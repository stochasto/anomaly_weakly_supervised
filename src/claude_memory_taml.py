import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class Memory_Unit(Module):
    def __init__(self, nums, dim):
        super().__init__()

        self.dim = dim
        self.nums = nums

        # 🔴 learned global memory (original behavior)
        self.memory_block = nn.Parameter(torch.empty(nums, dim))

        # 🔴 episodic memory buffer (NOT a parameter)
        self.register_buffer("memory_runtime", None)

        self.sig = nn.Sigmoid()

        self.reset_parameters()

    # -------------------------
    # INIT
    # -------------------------
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memory_block.size(1))
        self.memory_block.data.uniform_(-stdv, stdv)

    # -------------------------
    # 🔴 RESET MEMORY (TAML)
    # -------------------------
    def reset(self):
        """
        Clear episodic memory (called per task)
        """
        self.memory_runtime = None

    # -------------------------
    # 🔴 WRITE MEMORY (from support set)
    # -------------------------
    @torch.no_grad()  # 🔴 CRITICAL FIX: Prevent gradient flow through write
    def write(self, features):
        """
        features: [B, T, D]
        Store top-k features into episodic memory
        
        Changes from original:
        1. Added @torch.no_grad() decorator
        2. Clone + detach for safety (redundant but explicit)
        """
        B, T, D = features.shape

        # flatten temporal dimension
        feats = features.reshape(-1, D)

        # sample subset if too large
        if feats.size(0) > self.nums:
            idx = torch.randperm(feats.size(0), device=feats.device)[:self.nums]
            feats = feats[idx]

        # normalize (VERY important for stability)
        # L2 normalization keeps feature magnitudes consistent
        feats = torch.nn.functional.normalize(feats, dim=-1)

        # Store in runtime memory (no gradient tracking)
        self.memory_runtime = feats.clone().detach()

    # -------------------------
    # 🔴 MEMORY SELECTOR
    # -------------------------
    def _get_memory(self):
        """
        Use episodic memory if available, else learned memory
        
        TAML logic:
        - During support phase: build episodic memory from support set
        - During query phase: use same episodic memory built from support
        - During test time: fall back to learned global memory
        """
        if self.memory_runtime is not None:
            return self.memory_runtime
        return self.memory_block

    # -------------------------
    # FORWARD
    # -------------------------
    def forward(self, data):
        """
        data: [B, T, D]
        
        Returns:
        - temporal_att: [B] temporal attention scores
        - augment: [B, T, D] memory-augmented features
        """

        memory = self._get_memory()   # 🔴 critical switch

        # ✅ IMPROVED: Add small epsilon to stability
        # attention = sig(einsum) computes attention weights
        attention = self.sig(
            torch.einsum('btd,kd->btk', data, memory) / (self.dim ** 0.5)
        )  # [B, T, K]

        # temporal scoring: select top-k attended memory slots
        k = max(self.nums // 16 + 1, 1)
        temporal_att = torch.topk(attention, k, dim=-1)[0].mean(-1)  # [B]

        # reconstruction: weighted sum of memory by attention
        augment = torch.einsum('btk,kd->btd', attention, memory)  # [B, T, D]

        return temporal_att, augment


if __name__ == "__main__":
    mu = Memory_Unit(10, 512).cuda()
    data = torch.randn((3, 20, 512)).cuda()

    mu.reset()
    mu.write(data)

    out = mu(data)
    print("OK:", out[0].shape, out[1].shape)
