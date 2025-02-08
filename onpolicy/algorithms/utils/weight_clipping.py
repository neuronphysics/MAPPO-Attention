import torch
import torch.nn as nn
import math
from torch.optim.optimizer import Optimizer
from torch.nn.init import _calculate_fan_in_and_fan_out
from typing import Iterable, Optional, Tuple, Dict

import re
class InitBounds:
    def __init__(self):
        self.original_weights = {}
        self.original_layer_dims = {}

    def register_original_layer(self, name: str, weight: torch.Tensor):
        if name in self.original_weights:
            stored_weight = self.original_weights[name]
            if stored_weight is not None and not torch.equal(stored_weight, weight):
                raise ValueError(f"Conflicting weights for {name}")
     
        self.original_weights[name] = weight.detach()
        if weight.dim() == 2:
            in_dim, out_dim = weight.shape[1], weight.shape[0]
        elif weight.dim() == 4:
            in_dim = weight.shape[1] * weight.shape[2] * weight.shape[3]
            out_dim = weight.shape[0]
        else:
            return
        self.original_layer_dims[name] = (in_dim, out_dim)

    def get_bound(self, p: torch.Tensor, name: str) -> Optional[float]:
        # LoRA handling (unchanged)
        if "lora_" in name:
            base_name = name.split("lora_")[0]
            if base_name in self.original_layer_dims:
                in_dim, out_dim = self.original_layer_dims[base_name]
                if "lora_A" in name:
                    return 1.0 / math.sqrt(in_dim)
                elif "lora_B" in name:
                    return 1.0 / math.sqrt(p.size(1))
            return None

        # Original layer handling
        if p.dim() in (2, 4):  # Weights
            self.register_original_layer(name, p)
            fan_in, _ = _calculate_fan_in_and_fan_out(p)
            return 1.0 / math.sqrt(fan_in)
        elif p.dim() == 1:  # Bias
            base_name = name.replace(".bias", ".weight")
            if base_name in self.original_weights:
                fan_in, _ = _calculate_fan_in_and_fan_out(self.original_weights[base_name])
                return 1.0 / math.sqrt(fan_in)
        return None
       

class WeightClipping(Optimizer):
    """
    Enhanced optimizer wrapper that handles weight clipping for both regular and LoRA layers.
    """
    def __init__(
        self, 
        named_params: Iterable[Tuple[str, torch.nn.Parameter]],
        beta: float = 3.0,
        lora_beta: Optional[float] = None,  # Separate multiplier for LoRA layers
        optimizer_class: type = torch.optim.Adam,
        clip_last_layer: bool = True,
        **kwargs
    ):
        
        # Process parameters and detect hierarchy
        self.param_names={}
        params = []
        self.last_layer_names = []
        hierarchy = []

        for name, p in named_params:
            self.param_names[id(p)] = name
            params.append(p)
            
            # Only track original layers for hierarchy detection
            if p.dim() >= 2 and 'lora_' not in name:
                depth = self._calculate_depth(name)
                hierarchy.append((depth, name))

        # Identify last layers (deepest in hierarchy)
        if hierarchy:
            max_depth = max(d[0] for d in hierarchy)
            candidates = [name for depth, name in hierarchy if depth == max_depth]
            # Find weight-bias pairs
            weight_candidates = [n for n in candidates if n.endswith('.weight')]
            for weight_name in weight_candidates:
                bias_name = weight_name.replace('.weight', '.bias')
                if bias_name in self.param_names.values():
                   self.last_layer_names.extend([weight_name, bias_name])
                   break
        else:  # No bias found
            self.last_layer_names = [weight_candidates[-1]] if weight_candidates else []            

        # Create base optimizer
        defaults = dict(
            beta=beta,
            lora_beta=lora_beta or beta,
            clip_last_layer=clip_last_layer
        )
        super().__init__(params, defaults)
        self.optimizer = optimizer_class(self.param_groups, **kwargs)
        self.init_bounds = InitBounds()

        # Precompute clipping bounds
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                name = self.param_names[id(p)]
                bound = self.init_bounds.get_bound(p, name)
                if bound is None:
                    continue
                
                # Apply LoRA beta if applicable
                beta = group['lora_beta'] if 'lora_' in name else group['beta']
                p.clip_bound = bound * beta

    def _calculate_depth(self, name: str) -> int:
        # Count depth markers like '.layer.', '.block.', or numeric segments
        depth_markers = re.findall(r'\.(layer|block|stage)\d*\.|\d+', name)
        return len(depth_markers)

    def step(self, closure=None):
        """Perform optimization step with weight clipping"""
        self.optimizer.step(closure)
        self._clip_weights()

    def _clip_weights(self):
        """Apply weight clipping to all parameters"""
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue

                # Skip last layers if requested
                name = self.param_names.get(id(p), "")
                if not group['clip_last_layer'] and name in self.last_layer_names:
                    continue

                if hasattr(p, 'clip_bound'):
                    #p.data.clamp_(-p.clip_bound, p.clip_bound)
                    # Add gradual clipping for stability
                    bound = p.clip_bound
                    with torch.no_grad():
                        scale = torch.ones_like(p.data)
                        mask = p.data.abs() > bound
                        scale[mask] = bound / p.data.abs()[mask]
                        p.data.mul_(scale)
