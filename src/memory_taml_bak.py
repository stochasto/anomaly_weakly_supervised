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
    # 🔴 NEW: RESET MEMORY (TAML)
    # -------------------------
    def reset(self):
        """
        Clear episodic memory (called per task)
        """
        self.memory_runtime = None

    # -------------------------
    # 🔴 NEW: WRITE MEMORY (from support set)
    # -------------------------
    #@torch.no_grad()
    def write(self, features):
        """
        features: [B, T, D]
        Store top-k features into episodic memory
        """
        B, T, D = features.shape

        # flatten temporal dimension
        feats = features.reshape(-1, D)

        # sample subset if too large
        if feats.size(0) > self.nums:
            idx = torch.randperm(feats.size(0), device=feats.device)[:self.nums]
            feats = feats[idx]

        # normalize (VERY important for stability)
        feats = torch.nn.functional.normalize(feats, dim=-1)

        #self.memory_runtime = feats.detach()
        #Fix
        self.memory_runtime = feats.clone()

    # -------------------------
    # 🔴 NEW: MEMORY SELECTOR
    # -------------------------
    def _get_memory(self):
        """
        Use episodic memory if available, else learned memory
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
        """

        memory = self._get_memory()   # 🔴 critical switch

        # attention
        attention = self.sig(
            torch.einsum('btd,kd->btk', data, memory) / (self.dim ** 0.5)
        )  # [B, T, K]

        # temporal scoring
        k = max(self.nums // 16 + 1, 1)
        temporal_att = torch.topk(attention, k, dim=-1)[0].mean(-1)

        # reconstruction
        augment = torch.einsum('btk,kd->btd', attention, memory)

        return temporal_att, augment


if __name__ == "__main__":
    mu = Memory_Unit(10, 512).cuda()
    data = torch.randn((3, 20, 512)).cuda()

    mu.reset()
    mu.write(data)

    out = mu(data)
    print("OK:", out[0].shape, out[1].shape)