import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F


class Memory_Unit(Module):
    def __init__(self, nums, dim):
        super().__init__()

        self.dim = dim
        self.nums = nums

        # learned global memory (fallback)
        self.memory_block = nn.Parameter(torch.empty(nums, dim))

        # episodic runtime memory
        self.register_buffer("memory_runtime", None)

        self.reset_parameters()

    # -------------------------
    # INIT
    # -------------------------
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim)
        self.memory_block.data.uniform_(-stdv, stdv)

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):
        self.memory_runtime = None

    # -------------------------
    # WRITE
    # -------------------------
    @torch.no_grad()
    def write(self, features):
        """
        features: (B, T, D)
        """
        if features is None or features.numel() == 0:
            return

        if features.dim() != 3:
            raise ValueError(f"Expected (B,T,D), got {features.shape}")

        B, T, D = features.shape

        feats = features.reshape(-1, D)  # (B*T, D)

        # subsample if too large
        if feats.size(0) > self.nums:
            idx = torch.randperm(feats.size(0), device=feats.device)[:self.nums]
            feats = feats[idx]

        feats = F.normalize(feats, dim=-1)

        self.memory_runtime = feats.detach()

    # -------------------------
    # GET MEMORY
    # -------------------------
    def _get_memory(self):
        if self.memory_runtime is not None and self.memory_runtime.size(0) > 0:
            return self.memory_runtime
        return self.memory_block

    # -------------------------
    # FORWARD
    # -------------------------
    def forward(self, data):
        """
        data: (B, T, D)

        returns:
            temporal_att: (B, T)
            augment: (B, T, D)
        """

        if data.dim() != 3:
            raise ValueError(f"Expected (B,T,D), got {data.shape}")

        memory = self._get_memory()  # (K, D)
        K = memory.size(0)

        # -------------------------
        # ATTENTION
        # -------------------------
        logits = torch.einsum('btd,kd->btk', data, memory) / math.sqrt(self.dim)
        attention = torch.softmax(logits, dim=-1)  # (B, T, K)

        # -------------------------
        # SAFE TOP-K
        # -------------------------
        k = max(K // 16 + 1, 1)
        k = min(k, K)   # 🔴 CRITICAL FIX

        # (B, T, k)
        topk_vals = torch.topk(attention, k, dim=-1)[0]

        # temporal attention per timestep
        temporal_att = topk_vals.mean(dim=-1)  # (B, T)

        # -------------------------
        # AUGMENTATION
        # -------------------------
        augment = torch.einsum('btk,kd->btd', attention, memory)

        return temporal_att, augment


# -------------------------
# DEBUG
# -------------------------
if __name__ == "__main__":
    mu = Memory_Unit(10, 512).cuda()
    data = torch.randn((3, 20, 512)).cuda()

    mu.reset()
    mu.write(data)

    att, aug = mu(data)

    print("OK:", att.shape, aug.shape)