import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
# Assuming utils provides these implementations compatible with SlotAttention
from .utils import linear, gru_cell, PositionEmbed
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint


class FlashAttention(nn.Module):
    """
    FlashAttention module mimicking the structure of SlotAttention but using flash_attn_func.
    Maintains the iterative refinement loop with GRU and MLP updates.
    """
    def __init__(
        self,
        slot_size,        # Dimension of slots and features (embed_dim)
        mlp_size,         # Dimension of the hidden layer in the MLP update
        num_heads,        # Number of attention heads
        truncate,         # Gradient truncation method ('bi-level', 'fixed-point', 'none')
        drop_path=0.2,    # Stochastic depth dropout rate for the MLP update
        dropout_p=0.0,    # Dropout probability for the attention mechanism
    ):
        super().__init__()
        self.slot_size = slot_size
        self.num_heads = num_heads
        self.truncate = truncate
        self.dropout_p = dropout_p

        if slot_size % num_heads != 0:
             raise ValueError(f"slot_size ({slot_size}) must be divisible by num_heads ({num_heads})")
        self.head_dim = slot_size // num_heads

        self.norm_feature = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)

        # Linear projections for Query, Key, Value
        self.project_k = linear(slot_size, slot_size, bias=False)
        self.project_v = linear(slot_size, slot_size, bias=False)
        self.project_q = linear(slot_size, slot_size, bias=False)

        # GRU cell for iterative updates
        self.gru = gru_cell(slot_size, slot_size)

        # MLP for iterative updates
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_size, slot_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.truncate not in ['bi-level', 'fixed-point', 'none']:
             raise ValueError(f"Invalid truncate option: {self.truncate}")

    def forward(self, features, slots_init, num_iter=3):
        """
        Forward pass for FlashAttention.

        Args:
            features (Tensor): Input features, shape [B, num_features, slot_size].
            slots_init (Tensor): Initial slot states, shape [B, num_slots, slot_size].
            num_iter (int): Number of attention iterations.

        Returns:
            Tuple[Tensor, None]: Updated slots tensor and None (as flash_attn doesn't return attn weights).
        """
        # Normalize features and project to Key and Value
        # `features` shape: [B, num_features, slot_size]
        features_norm = self.norm_feature(features)
        k = self.project_k(features_norm)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features_norm)  # Shape: [B, num_features, slot_size]

        B, N_feat, D = features.shape
        _, N_slots, _ = slots_init.shape

        slots = slots_init # Shape: [B, num_slots, slot_size]

        # Multiple rounds of attention and updates
        for i in range(num_iter):
            # Apply gradient truncation based on the specified method
            if i == num_iter - 1:
                if self.truncate == 'bi-level':
                    # Detach previous slots but allow gradients through initial slots
                    slots = slots.detach() + slots_init - slots_init.detach()
                elif self.truncate == 'fixed-point':
                    # Detach previous slots completely
                    slots = slots.detach()

            slots_prev = slots
            # Normalize slots and project to Query
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm) # Shape: [B, num_slots, slot_size]

            # Reshape Q, K, V for FlashAttention: (B, Seq_len, Num_heads, Head_dim)
            q_reshaped = q.view(B, N_slots, self.num_heads, self.head_dim)
            k_reshaped = k.view(B, N_feat, self.num_heads, self.head_dim)
            v_reshaped = v.view(B, N_feat, self.num_heads, self.head_dim)

            # Compute attention updates using flash_attn_func
            # Output shape: (B, N_slots, num_heads, head_dim)
            updates_reshaped = flash_attn_func(
                q_reshaped, k_reshaped, v_reshaped,
                dropout_p=self.dropout_p,
                softmax_scale=None, # Defaults to 1/sqrt(head_dim)
                causal=False       # Non-causal attention
            )

            # Reshape updates back to (B, N_slots, D)
            updates = updates_reshaped.view(B, N_slots, D)

            # Update slots using GRU and MLP with gradient checkpointing
            slots = checkpoint(self._update_slots, updates, slots_prev, use_reentrant=False)

        # Note: flash_attn_func does not return attention weights.
        # Returning None to maintain API structure.
        attn = None
        return slots, attn

    def _update_slots(self, updates, slots_prev):
        """Applies the GRU and MLP update to the slots."""
        B, N_slots, D = updates.shape
        # GRU update
        slots = self.gru(
            updates.reshape(-1, D),
            slots_prev.reshape(-1, D)
        ).reshape(B, N_slots, D)
        # MLP update with DropPath and residual connection
        slots = slots + self.drop_path(self.mlp(self.norm_mlp(slots)))
        return slots


class FlashAttentionEncoder(nn.Module):
    """
    Encoder module using the FlashAttention layer.
    Mirrors the structure of SlotAttentionEncoder.
    """
    def __init__(
            self,
            num_iter,         # Number of attention iterations
            num_slots,        # Number of slots
            feature_size,     # Input feature dimension (before MLP projection)
            slot_size,        # Dimension of slots (embed_dim for attention)
            mlp_size,         # Hidden layer size in update MLP and initial feature MLP
            num_heads,        # Number of attention heads
            resolution,       # Input feature map resolution (e.g., [H, W])
            truncate='bi-level', # Gradient truncation method
            init_method='embedding', # Slot initialization method
            drop_path=0.2,    # Stochastic depth rate for update MLP
            dropout_p=0.0     # Attention dropout probability
        ):
        super().__init__()

        self.num_iter = num_iter
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.init_method = init_method
        self.num_heads = num_heads

        # Positional embedding for input features
        self.pos_emb = PositionEmbed(feature_size, resolution)

        # MLP to process input features before attention
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, slot_size) # Project to slot_size
        )

        # Instantiate the FlashAttention module
        self.flash_attention = FlashAttention(
            slot_size=slot_size,
            mlp_size=mlp_size, # MLP size for the update step inside FlashAttention
            num_heads=num_heads,
            truncate=truncate,
            drop_path=drop_path,
            dropout_p=dropout_p
        )

        # Slot initialization parameters/layers
        if init_method not in ['shared_gaussian', 'embedding']:
             raise ValueError(f"Invalid init_method: {init_method}")
        self.init_method = init_method
        if init_method == 'shared_gaussian':
            # Learnable mean and log-sigma for Gaussian initialization
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)
        elif init_method == 'embedding':
            # Learnable embedding for each slot
            self.slots_init_embedding = nn.Embedding(num_slots, slot_size)
            nn.init.xavier_uniform_(self.slots_init_embedding.weight)


    def forward(self, f, sigma=None, slots_init=None):
        """
        Forward pass for the FlashAttentionEncoder.

        Args:
            f (Tensor): Input feature map, shape [B, C, H, W].
            sigma (float, optional): Noise scale for embedding initialization. Defaults to None.
            slots_init (Tensor, optional): Pre-initialized slots. Defaults to None.

        Returns:
            dict: Dictionary containing 'slots', 'slots_init', and 'attn' (which is None).
        """
        B, C, H, W = f.shape

        # Apply positional embedding and flatten spatial dimensions
        f_pos = self.pos_emb(f) # Shape [B, C, H, W]
        f_flat = f_pos.flatten(start_dim=2).permute(0, 2, 1) # Shape [B, H*W, C]

        # Process features with MLP
        features_processed = self.feature_mlp(f_flat) # Shape [B, H*W, slot_size]

        # Initialize slots if not provided
        if slots_init is None:
            if self.init_method == 'shared_gaussian':
                if sigma is None: # Use learned sigma if sigma not provided
                    sigma = torch.exp(self.slot_log_sigma)
                slots_init = torch.randn(B, self.num_slots, self.slot_size).type_as(f) * sigma + self.slot_mu
            elif self.init_method == 'embedding':
                mu = self.slots_init_embedding.weight.expand(B, -1, -1)
                if sigma is None: # Default sigma if not provided
                    sigma = 1.0
                # Add noise scaled by sigma and detached mu
                z = torch.randn_like(mu).type_as(f)
                slots_init = mu + z * sigma * mu.detach()

        # Define function for checkpointing the main flash_attention computation
        def run_flash_attention(features_inner, slots_init_inner):
            return self.flash_attention(features_inner, slots_init_inner, self.num_iter)

        # Run FlashAttention with gradient checkpointing
        slots, attn = checkpoint(run_flash_attention, features_processed, slots_init, use_reentrant=False)

        return {
            'slots': slots,        # Final slots [B, num_slots, slot_size]
            'slots_init': slots_init, # Initial slots used [B, num_slots, slot_size]
            'attn': attn,          # Attention weights (None for FlashAttention)
        }
