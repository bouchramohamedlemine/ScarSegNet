
import torch
from torch import nn

class T_Max_Avg_pooling(nn.Module):
    def __init__(self, input_size, init_T, k_ratio):
        super().__init__()
        self.k = max(1, int(input_size * input_size * k_ratio))
        self.T = nn.Parameter(torch.tensor(init_T, dtype=torch.float32))  # learnable (pre-sigmoid)
        self.tau = 0.1  # temperature for STE

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)

        # top-K per channel
        topk_values, _ = torch.topk(x_flat, self.k, dim=2)

        # normalize per channel
        topk_normalized = topk_values / (topk_values.max(dim=2, keepdim=True).values + 1e-6)

        # stats
        max_vals = topk_values.max(dim=2).values
        avg_vals = topk_values.mean(dim=2)

        # learnable threshold in (0,1)
        T = torch.sigmoid(self.T)   

        # STRICT decision with straight-through estimator (STE)
        s = topk_normalized.amin(dim=2)                  # min ≡ all(topk_norm >= T)
        logits = (s - T) / self.tau
        gate_soft = torch.sigmoid(logits)                # For gradients, we keep gate smooth  
        gate_hard = (logits >= 0).float()                # For forward pass, we use strict 0/1  
        gate = gate_hard.detach() - gate_soft.detach() + gate_soft  # STE

        # Decide whether to apply avg pooling or max pooling, 
        # however, in backward pass to make t learnable we are applying a combiantion of both as the gate is not strict. 
        pooled = gate * max_vals + (1 - gate) * avg_vals

        return pooled