import torch
import math

class InitBounds:
    """Modified to handle all parameter types gracefully"""
    def __init__(self):
        self.previous_weight = None

    def get(self, p):
        try:
            if p.dim() == 1:
                if self.previous_weight is None:
                    return None
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.previous_weight)
                return 1.0 / math.sqrt(fan_in)
            elif p.dim() in (2, 4):
                self.previous_weight = p
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(p)
                return 1.0 / math.sqrt(fan_in)
            
        except Exception as e:
            print(f"Error calculating initialization bounds: {str(e)}")
            print(f"Parameter shape: {p.shape}, dim: {p.dim()}")
            return None

class WeightClipping(torch.optim.Optimizer):
    def __init__(self, named_params, beta=1.0, optimizer=torch.optim.Adam, **kwargs):
        # Store parameter names and filter targets
        named_params = list(named_params)
        self.param_names = {id(p): name for name, p in named_params}
        params = [p for name, p in named_params]
        # Validate parameters exist
        if not params:
            raise ValueError("No parameters found for optimization. Check if slot_attn/LoRA layers are properly unfrozen.")
        
        defaults = dict(beta=beta)
        super().__init__(params, defaults)
        
        self.optimizer = optimizer(self.param_groups, **kwargs)
        self.init_bounds = InitBounds()

    def step(self):
        self.optimizer.step()
        self.weight_clipping()

    def weight_clipping(self):
        for group in self.param_groups:
            beta = group['beta']
            for p in group["params"]:
                # Get parameter metadata
                name = self.param_names.get(id(p), "")
                is_slot_attn = "slot_attn" in name
                is_lora = "lora_" in name
                requires_grad = p.requires_grad

                # Compute bounds for all parameters to maintain state
                bound = self.init_bounds.get(p)
                
                # Apply clipping only to target parameters
                if bound and ( (is_slot_attn and is_lora) or (is_slot_attn and requires_grad) ):
                    with torch.no_grad():
                          p.data.clamp_(-beta * bound, beta * bound)