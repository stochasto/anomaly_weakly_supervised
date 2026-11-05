# PATCHED Memory_Unit class
# Replace your existing Memory_Unit in model_min_upgrade.py or memory.py with this:

import torch
import torch.nn as nn

class Memory_Unit(nn.Module):
    """
    Improved memory unit with:
    - Better initialization (normalized + orthogonal-like)
    - Temperature for stable attention
    - Proper normalization to prevent attention collapse
    """
    def __init__(self, nums, dim):
        super().__init__()
        self.nums = nums
        self.dim = dim
        
        # ===== CHANGE 4: Better initialization =====
        # Initialize with normalized random vectors
        memory = torch.randn(nums, dim)
        memory = memory / (memory.norm(dim=1, keepdim=True) + 1e-8)
        self.memory = nn.Parameter(memory)
        
        # Add temperature parameter for stable attention softmax
        # Prevents attention from becoming too sharp (collapse) or too flat
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        # x: [B, T, D]
        # Normalize features for stable attention
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        mem_norm = self.memory / (self.memory.norm(dim=1, keepdim=True) + 1e-8)
        
        # Scaled dot product attention with temperature
        # Temperature prevents attention from exploding or vanishing
        attn = torch.einsum('btd,nd->btn', x_norm, mem_norm) / (self.temperature.abs() + 0.1)
        attn = torch.softmax(attn, dim=-1)
        
        # Read from memory
        aug = torch.einsum('btn,nd->btd', attn, mem_norm)
        return attn.mean(-1), aug


# =============================================================================
# USAGE IN model_min_upgrade.py:
# =============================================================================
# 
# Instead of:
#   self.Amemory = Memory_Unit(nums=a_nums, dim=512)
#   self.Nmemory = Memory_Unit(nums=n_nums, dim=512)
#
# Just use the same code - but import from the patched memory module.
#
# If you have custom memory module, replace the Memory_Unit class with above.
# =============================================================================
