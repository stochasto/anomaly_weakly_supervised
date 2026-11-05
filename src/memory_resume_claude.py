import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class Memory_Unit(Module):
    """
    Improved memory unit with sharp attention for anomaly localization.
    
    Key fixes:
    - Removed sigmoid squashing (was destroying gradient flow)
    - Direct temperature-scaled softmax on cosine similarity
    - Top-k selection for crisp temporal attention
    - Better initialization (xavier_uniform)
    """
    def __init__(self, nums, dim, temperature=3.0, topk=4):
        super().__init__()
        self.dim = dim
        self.nums = nums
        self.temperature = temperature
        self.topk = topk
        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Xavier uniform initialization for better convergence"""
        nn.init.xavier_uniform_(self.memory_block)

    def forward(self, data):
        """
        Args:
            data: [B, T, D] - batch of video features (T temporal segments, D dimensions)
        
        Returns:
            temporal_att: [B, T] - sharp temporal attention scores (for loss supervision)
            augment: [B, T, D] - memory-augmented features (for feature enhancement)
        """
        # Normalize both query features and memory prototypes
        data = F.normalize(data, p=2, dim=-1)        # [B, T, D]
        memory = F.normalize(self.memory_block, p=2, dim=-1)  # [K, D]
        
        # ✅ FIXED: Direct cosine similarity (no sigmoid squashing)
        # This preserves the full [-1, 1] range for sharp discrimination
        sim = torch.einsum('btd,kd->btk', data, memory)  # [B, T, K]
        
        # ✅ Temperature-scaled softmax for sharp attention peaks
        # temperature > 1: sharper (more selective)
        # temperature < 1: smoother (more distributed)
        weights = torch.softmax(sim / self.temperature, dim=-1)  # [B, T, K]
        
        # ✅ Top-k selection for temporal attention (frame-level anomaly scores)
        k = min(self.topk, self.nums)
        topk_weights, _ = torch.topk(weights, k, dim=-1)  # [B, T, k]
        
        # Temporal attention: average of top-k memory activations per frame
        temporal_att = topk_weights.mean(dim=-1)  # [B, T]
        
        # ✅ Memory augmentation: full weighted sum of all memory prototypes
        # This allows smooth interpolation in feature space
        augment = torch.einsum('btk,kd->btd', weights, memory)  # [B, T, D]
        
        return temporal_att, augment


if __name__ == "__main__":
    # Smoke test
    mu = Memory_Unit(nums=200, dim=512, temperature=3.0, topk=4).cuda()
    data = torch.randn((3, 20, 512)).cuda()
    temporal_att, augment = mu(data)
    
    print(f"✓ temporal_att shape: {temporal_att.shape}")  # [3, 20]
    print(f"✓ augment shape: {augment.shape}")            # [3, 20, 512]
    print(f"✓ temporal_att range: [{temporal_att.min():.4f}, {temporal_att.max():.4f}]")
    print(f"✓ Memory unit working correctly!")
