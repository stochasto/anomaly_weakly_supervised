import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class Memory_Unit(Module):
    def __init__(self, nums, dim, temperature=3.0, topk=4):
        super().__init__()
        self.dim = dim
        self.nums = nums
        self.temperature = temperature
        self.topk = topk
        self.sig = nn.Sigmoid()

        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Better initialization than tiny uniform
        nn.init.xavier_uniform_(self.memory_block)

    #def forward(self, data):
    #    """
    #    data: [B, T, D]
    #    returns:
    #        temporal_att: [B, T]
    #        augment:      [B, T, D]
    #    """

    #    # Normalize both query features and memory slots
    #    data = F.normalize(data, p=2, dim=-1)
    #    memory = F.normalize(self.memory_block, p=2, dim=-1)

    #    # Similarity scores
    #    logits = torch.einsum('btd,kd->btk', data, memory)

    #    # Sharpened competitive attention
    #    attention = torch.softmax(logits * self.temperature, dim=-1)

    #    # Sharp temporal attention for loss supervision
    #    k = min(self.topk, self.nums)
    #    topk_vals, _ = torch.topk(attention, k, dim=-1)
    #    temporal_att = topk_vals.mean(dim=-1)   # [B, T]

    #    # Full differentiable memory augmentation
    #    augment = torch.einsum('btk,kd->btd', attention, memory)

    #    return temporal_att, augment
    
    #def forward(self, data):
    #    data  = F.normalize(data, p=2, dim=-1)
    #    memory = F.normalize(self.memory_block, p=2, dim=-1)

    #    # Cosine similarity, scaled gently to avoid saturation
    #    sim = torch.einsum('btd,kd->btk', data, memory)       # [-1, 1]
    #    raw_att = torch.sigmoid(sim * 2.0)                    # keeps some spread

    #    # Differentiable softmax with temperature
    #    weights = torch.softmax(raw_att * self.temperature, dim=-1)
    #    temporal_att = (weights * raw_att).sum(dim=-1)

    #    # Augment: convex combination of memory prototypes
    #    augment = torch.einsum('btk,kd->btd', weights, memory)

    #    return temporal_att, augment
    
    #def forward(self, data):  ####data size---> B,T,D       K,V size--->K,D
    #    attention = self.sig(torch.einsum('btd,kd->btk', data, self.memory_block) / (self.dim**0.5))   #### Att---> B,T,K
    #    temporal_att = torch.topk(attention, self.nums//16+1, dim = -1)[0].mean(-1)
    #    augment = torch.einsum('btk,kd->btd', attention, self.memory_block)                   #### feature_aug B,T,D
    #    return temporal_att, augment
    
    def forward(self, data):
        data = F.normalize(data, p=2, dim=-1)
        memory = F.normalize(self.memory_block, p=2, dim=-1)

        sim = torch.einsum('btd,kd->btk', data, memory)

        raw_att = torch.sigmoid(sim * 2.0)               # broad but meaningful spread
        weights = torch.softmax(raw_att * 2.0, dim=-1)   # differentiable mild sharpening
        temporal_att = (weights * raw_att).sum(dim=-1)
        augment = torch.einsum('btk,kd->btd', raw_att, memory)   # broad gradient

        return temporal_att, augment


if __name__ == "__main__":
    mu = Memory_Unit(nums=200, dim=512).cuda()
    data = torch.randn((3, 20, 512)).cuda()
    temporal_att, augment = mu(data)

    print("temporal_att:", temporal_att.shape)  # [3, 20]
    print("augment:", augment.shape)            # [3, 20, 512]