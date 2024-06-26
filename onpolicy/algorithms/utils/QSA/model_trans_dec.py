import os
import sys
from .utils import *
from .dvae import dVAE
from .slot_attn import SlotAttentionEncoder
from .encoder import Encoder
from .transformer import PositionalEncoding, TransformerDecoder
from sklearn.cluster import KMeans


class SLATE(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.attention_module == 'RIM':
            num_slot = args.rim_num_units
        else:
            num_slot = args.scoff_num_units
        slot_dim = args.hidden_size // num_slot

        self.vocab_size = args.vocab_size
        self.d_model = args.d_model

        self.dvae = dVAE(args.vocab_size, args.img_channels, args.dvae_kernel_size)
        N_tokens = (args.crop_size // 4) * (args.crop_size // 4)

        self.positional_encoder = PositionalEncoding(1 + N_tokens, args.d_model, args.dropout)

        self.slot_attn = SlotAttentionEncoder(
            args.num_iter, num_slot, args.feature_size,
            slot_dim, args.mlp_size,
            (args.crop_size, args.crop_size), args.truncate,
            args.init_method, args.drop_path)

        self.dictionary = OneHotDictionary(args.vocab_size + 1, args.d_model)
        self.slot_proj = linear(slot_dim, args.d_model, bias=False)

        self.tf_dec = TransformerDecoder(
            args.num_dec_blocks, N_tokens, args.d_model, args.num_heads, args.dropout)

        self.out = linear(args.d_model, args.vocab_size, bias=False)

        self.backbone = Encoder(args.encoder_channels, args.encoder_strides, args.encoder_kernel_size)

        self.ortho_loss_fn = PerpetualOrthogonalProjectionLoss(num_classes=num_slot, feat_dim=slot_dim)

        self.use_post_cluster = args.use_post_cluster
        self.lambda_c = args.lambda_c
        self.num_slots = num_slot
        self.slot_size = slot_dim
        if self.use_post_cluster:
            self.register_buffer('post_cluster', torch.zeros(1, num_slot, slot_dim))
            nn.init.xavier_normal_(self.post_cluster)
        self.kmeans = KMeans(n_clusters=num_slot, random_state=args.seed) if args.use_kmeans else None

    def forward(self, image, visualize=False, tau=0.1, sigma=0, is_Train=False):
        """
        image: batch_size x img_channels x H x W
        """
        out = {}
        loss = {}

        B, C, H, W = image.size()

        # dvae encode
        mse, recon, z = self.dvae(image, tau, return_z=True)
        loss['mse'] = mse

        # target tokens for transformer
        target = z.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B x H_z*W_z x C

        # add BOS token
        input = torch.cat([torch.zeros_like(target[..., :1]), target], dim=-1)  # B x H_z*W_z x C+1
        input = torch.cat([torch.zeros_like(input[..., :1, :]), input], dim=-2)  # B x 1 + H_z*W_z x C+1
        input[:, 0, 0] = 1.0

        # tokens to embeddings
        emb_input = self.positional_encoder(self.dictionary(input))

        # apply slot attention
        f = self.backbone(image)
        if self.use_post_cluster:
            slots_init = self.post_cluster.repeat(B, 1, 1)
            slot_attn_out = self.slot_attn(f, sigma=sigma, slots_init=slots_init)
            slots = slot_attn_out['slots']
            if is_Train:
                # update post cluster, shape: 1 x num_slots x slot_size
                if self.kmeans is not None:
                    self.kmeans.fit(slots.detach().reshape(-1, self.slot_size).cpu().numpy())
                    update = torch.Tensor(self.kmeans.cluster_centers_.reshape(1, self.num_slots, self.slot_size)).to(
                        image.device)
                    self.post_cluster = self.lambda_c * update + (1 - self.lambda_c) * self.post_cluster
                else:
                    update = slots.detach().mean(dim=0, keepdim=True)
                    self.post_cluster = self.lambda_c * update + (1 - self.lambda_c) * self.post_cluster
        else:
            slot_attn_out = self.slot_attn(f, sigma=sigma)
            slots = slot_attn_out['slots']
        attns = slot_attn_out['attn']
        attns = attns.reshape(B, -1, 1, H, W)

        out['slots'] = slots
        labels = torch.arange(self.num_slots).unsqueeze(0).repeat(B, 1).reshape(-1).to(slots.device)
        out['sim_loss'] = self.ortho_loss_fn(slots.reshape(-1, self.slot_size), labels)

        # apply transformer
        slots_t = self.slot_proj(slots)
        decoder_output = self.tf_dec(emb_input[:, :-1], slots_t)
        pred = self.out(decoder_output)

        cross_entropy = -(target * torch.log_softmax(pred, dim=-1)).sum() / B

        loss["cross_entropy"] = cross_entropy

        if visualize:
            with torch.no_grad():
                out['recon'] = recon
                pred_z = F.one_hot(pred.argmax(dim=-1), self.vocab_size).float().transpose(1, 2).reshape(B,
                                                                                                         self.vocab_size,
                                                                                                         H // 4, W // 4)
                pred_image = self.dvae.decoder(pred_z)
                out["pred_image"] = pred_image.clamp(0, 1)

        out['attns'] = attns
        out['loss'] = loss
        return out


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs

    def get_embedding(self):
        return self.dictionary.weight


class PerpetualOrthogonalProjectionLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2048, no_norm=False, use_attention=True):
        super(PerpetualOrthogonalProjectionLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.no_norm = no_norm
        self.use_attention = use_attention

        self.class_centres = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, features, labels=None):

        if self.use_attention:
            features_weights = torch.matmul(features, features.T)
            features_weights = F.softmax(features_weights, dim=1)
            features = torch.matmul(features_weights, features)

        #  features are normalized
        if not self.no_norm:
            features = F.normalize(features, p=2, dim=1)
        normalized_class_centres = F.normalize(self.class_centres, p=2, dim=1)

        labels = labels[:, None]  # extend dim
        class_range = torch.arange(self.num_classes, device=self.device).long()
        class_range = class_range[:, None]  # extend dim
        label_mask = torch.eq(labels, class_range.t()).float().to(self.device)
        feature_centre_variance = torch.matmul(features, normalized_class_centres.t())
        same_class_loss = (label_mask * feature_centre_variance).sum() / (label_mask.sum() + 1e-6)
        diff_class_loss = ((1 - label_mask) * feature_centre_variance).sum() / ((1 - label_mask).sum() + 1e-6)

        loss = 0.5 * (1.0 - same_class_loss) + torch.abs(diff_class_loss)

        return loss
