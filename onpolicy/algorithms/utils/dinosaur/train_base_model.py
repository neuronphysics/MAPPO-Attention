import argparse
import functools

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from onpolicy.algorithms.utils.dinosaur.ocl.feature_extractors.timm import TimmFeatureExtractor
from onpolicy.algorithms.utils.dinosaur.ocl.conditioning import RandomConditioning
from onpolicy.algorithms.utils.dinosaur.ocl.perceptual_grouping import SlotAttentionGrouping
from onpolicy.algorithms.utils.dinosaur.ocl.neural_networks.wrappers import Sequential
from onpolicy.algorithms.utils.dinosaur.ocl.neural_networks.convenience import build_two_layer_mlp, \
    build_transformer_decoder
from onpolicy.algorithms.utils.dinosaur.ocl.decoding import AutoregressivePatchDecoder
from onpolicy.algorithms.utils.dinosaur.ocl.neural_networks.positional_embedding import DummyPositionEmbed
from torch import nn
from torch.optim import Adam
import torch
import torchvision.utils as vutils
from onpolicy.algorithms.utils.dinosaur.ocl.losses import ReconstructionLoss
from onpolicy.algorithms.utils.dinosaur.ocl.scheduling import exponential_decay_after_optional_warmup
from onpolicy.algorithms.utils.dinosaur.data_loader import GlobDataset
import numpy as np
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self,
                 feature_extractor: nn.Module,
                 conditioning: nn.Module,
                 perceptual_grouping: nn.Module,
                 decoder: nn.Module
                 ):
        super(BaseModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.conditioning = conditioning
        self.perceptual_grouping = perceptual_grouping
        self.decoder = decoder

    def forward(self, input_data):
        b, c, h, w = input_data.shape
        feature_out = self.feature_extractor(input_data)
        con_out = self.conditioning(b)

        perceptual_grouping_out = self.perceptual_grouping(feature_out, con_out)
        decode_out = self.decoder(object_features=perceptual_grouping_out.objects,
                                  target=feature_out.features,
                                  image=input_data,
                                  masks=perceptual_grouping_out.feature_attributions,
                                  )
        return decode_out, perceptual_grouping_out


def generate_model(args):
    if args.attention_module == 'RIM':
        num_slot = args.rim_num_units
    else:
        num_slot = args.scoff_num_units
    slot_dim = args.hidden_size // num_slot

    feature_extractor_conf = {
        "model_name": "vit_base_patch14_dinov2.lvd142m",
        "pretrained": False,
        "freeze": True,
        "feature_level": 12,
    }

    feature_extractor = TimmFeatureExtractor(**feature_extractor_conf)

    cond_conf = {
        "object_dim": slot_dim,
        "n_slots": num_slot
    }
    conditioning = RandomConditioning(**cond_conf)

    # perceptual_grouping
    input_feature_dim = 768  # TODO we doubled it
    positional_embedding_layer_1 = DummyPositionEmbed()

    positional_embedding_layer_2_conf = {
        "input_dim": input_feature_dim,
        "output_dim": slot_dim,
        "hidden_dim": input_feature_dim,
        "initial_layer_norm": True
    }
    positional_embedding_layer_2 = build_two_layer_mlp(**positional_embedding_layer_2_conf)

    positional_embedding = Sequential(*[positional_embedding_layer_1, positional_embedding_layer_2])

    ff_mlp_conf = {
        "input_dim": slot_dim,
        "output_dim": slot_dim,
        "hidden_dim": slot_dim * 4,
        "initial_layer_norm": True,
        "residual": True,
    }
    ff_mlp = build_two_layer_mlp(**ff_mlp_conf)

    perceptual_grouping_conf = {
        "feature_dim": slot_dim,
        "object_dim": slot_dim,
        "use_projection_bias": False,
        "positional_embedding": positional_embedding,
        "ff_mlp": ff_mlp
    }
    perceptual_grouping = SlotAttentionGrouping(**perceptual_grouping_conf)

    # decoder
    decoder_bb_conf = {
        "n_layers": 4,
        "n_heads": 8,
        "return_attention_weights": True
    }
    decoder_bb = functools.partial(build_transformer_decoder, **decoder_bb_conf)

    decoder_conf = {
        "object_dim": slot_dim,
        "output_dim": input_feature_dim,
        "num_patches": 100,
        "decoder_cond_dim": input_feature_dim,
        "use_input_transform": True,
        "use_decoder_masks": True,
        "decoder": decoder_bb,
    }

    decoder = AutoregressivePatchDecoder(**decoder_conf)

    model = BaseModel(feature_extractor, conditioning, perceptual_grouping, decoder)

    model.to(args.device)
    return model


def get_loss_function(args):
    loss_fn_conf = {
        "loss_type": "l1"
    }
    loss_fn = ReconstructionLoss(**loss_fn_conf)
    return loss_fn


def get_optimizer(args, model):
    optimizer = Adam(model.parameters(), lr=args.slot_att_lr)

    scheduler = exponential_decay_after_optional_warmup(optimizer, decay_rate=0.5, decay_steps=100000,
                                                        warmup_steps=10000, )
    return optimizer, scheduler


def train_dino(args):
    train_dataset = GlobDataset(root=args.slot_att_work_path + "world_data/*", phase='train', img_glob="*.pt",
                                crop_repeat=args.slot_att_crop_repeat, crop_size=args.slot_att_crop_size)
    # val_dataset = GlobDataset(root=args.slot_att_work_path + "world_data/*", phase='val', img_glob="*.pt",
    #                           seq_len=args.seq_len)

    loader_kwargs = {
        'batch_size': args.slot_train_batch,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': True,
    }

    train_loader = DataLoader(train_dataset, sampler=None, **loader_kwargs)
    # val_loader = DataLoader(val_dataset, sampler=None, **loader_kwargs)

    writer = SummaryWriter(args.slot_att_work_path + "tensorboard/")

    model: nn.Module = generate_model(args)

    loss_fn = get_loss_function(args)
    optimizer, scheduler = get_optimizer(args, model)

    model.train()
    for ep in tqdm(range(args.slot_train_ep)):
        for idx, batch_data in enumerate(train_loader):
            # batch, channel, height, width
            out, per_out = model(batch_data.to(args.device))

            loss = loss_fn(out.reconstruction, out.target) + slot_similarity_loss(per_out.objects)

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.slot_clip_grade_norm)
            optimizer.step()

            writer.add_scalar("train_loss", loss, ep * len(train_loader) + idx)

        if ep % args.slot_log_fre == 0:
            images, combined_mask = visualize_img(out, batch_data)
            writer.add_image("masked image", images, global_step=ep)
            writer.add_image("masks", combined_mask, global_step=ep)

        if ep % args.slot_save_fre == 0:
            save_slot_att_model(model, args)


def slot_similarity_loss(slots):
    """
    Calculate the similarity loss for slots with shape (batch, num_slot, hidden_size).
    """
    batch_size, num_slots, slot_dim = slots.shape
    # Normalize slot features
    slots = F.normalize(slots, dim=-1)  # Normalize along the hidden_size dimension

    # Randomly permute the slots
    perm = torch.randperm(slots.size(1)).to(slots.device)  # Permute along the num_slot dimension

    # Select a subset of n slots
    selected_slots = slots[:, perm[:num_slots], :]  # [batch, n, hidden_size]

    # Compute similarity matrix
    sim_matrix = torch.bmm(selected_slots, selected_slots.transpose(1, 2)) * (
            1 / np.sqrt(slots.size(2)))  # [batch, n, n]

    # Create mask to remove diagonal elements (self-similarity)
    mask = torch.eye(num_slots).to(slots.device).repeat(batch_size, 1, 1)  # [1, n, n]

    # Mask out the diagonal elements
    sim_matrix = sim_matrix - mask * sim_matrix

    # Compute similarity loss
    sim_loss = sim_matrix.sum(dim=(1, 2)) / (num_slots * (num_slots - 1))

    return sim_loss.mean()  # Return the mean similarity loss over the batch


def visualize_img(out, original):
    B, C, H, W = original.shape
    img_mask = out.masks_as_image.unsqueeze(2)[0].cpu()
    S, _, _, _ = img_mask.shape
    slot_img = original.unsqueeze(1)[0]  # slot, channel, H, W

    masked_image = img_mask * slot_img  # slot, channel, H, W
    masked_image = masked_image.permute(1, 2, 0, 3).reshape(C, H, -1)
    masked_image = torch.cat([original[0], masked_image], dim=-1)

    combined_mask = img_mask.permute(1, 2, 0, 3).reshape(1, H, -1)
    return masked_image, combined_mask


def visualize_video(out, original, N=1, num_slots=8):
    B, T, H, W, C = original.size()
    video = original.permute(0, 1, 4, 2, 3)
    recon = out['outputs']['video'].reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).cpu()
    att_mask = out['outputs']['targets_masks'].reshape(B, T, -1, H, W, 1).permute(0, 1, 2, 5, 3, 4).cpu()
    attns_vis = out['outputs']['video_channels'].reshape(B, T, -1, H, W, C).permute(0, 1, 2, 5, 3, 4).cpu()

    frames = []
    for t in range(T):
        video_t = video[:N, t, None, :, :, :]
        recon_t = recon[:N, t, None, :, :, :]
        att_mask_t = att_mask[:N, t, :, :, :, :].repeat(1, 1, 3, 1, 1)
        attns_vis_t = attns_vis[:N, t, :, :, :, :]

        total_mask = att_mask_t.sum(dim=1)
        total_mask = total_mask.unsqueeze(1)
        total_mask[total_mask > 1] = 1
        total_mask_video = video_t * total_mask

        # tile
        tiles = torch.cat((video_t, recon_t, total_mask_video, att_mask_t, attns_vis_t), dim=1).flatten(end_dim=1)

        # grid
        frame = vutils.make_grid(tiles, nrow=(num_slots + 3), pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames


def load_slot_att_model(model, args):
    if args.attention_module == "RIM":
        num_slots = args.rim_num_units
    else:
        num_slots = args.scoff_num_units
    latent_size = args.hidden_size // num_slots
    model_name = "ns_" + str(num_slots) + "_ls_" + str(latent_size) + "_model.pt"
    model_state_dict = torch.load(args.slot_att_work_path + model_name)
    model.load_state_dict(model_state_dict)


def save_slot_att_model(model, args):
    if args.attention_module == "RIM":
        num_slots = args.rim_num_units
    else:
        num_slots = args.scoff_num_units
    latent_size = args.hidden_size // num_slots
    model_name = "ns_" + str(num_slots) + "_ls_" + str(latent_size) + "_model.pt"
    # tmp = {}
    # tmp['conditioning'] = model.conditioning.mo
    # tmp['perceptual_grouping'] = model.perceptual_grouping
    # tmp['decoder'] = model.decoder
    torch.save(model.state_dict(), args.slot_att_work_path + model_name)
