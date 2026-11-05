import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module

class Memory_Unit(Module):
    def __init__(self, nums, dim):
        super().__init__()
        self.dim = dim
        self.nums = nums
        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.sig = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memory_block.size(1))
        self.memory_block.data.uniform_(-stdv, stdv)
        if self.memory_block is not None:
            self.memory_block.data.uniform_(-stdv, stdv)
    
       
    #def forward(self, data):  ####data size---> B,T,D       K,V size--->K,D
    #    attention = self.sig(torch.einsum('btd,kd->btk', data, self.memory_block) / (self.dim**0.5))   #### Att---> B,T,K
    #    temporal_att = torch.topk(attention, self.nums//16+1, dim = -1)[0].mean(-1)
    #    augment = torch.einsum('btk,kd->btd', attention, self.memory_block)                   #### feature_aug B,T,D
    #    return temporal_att, augment
    
    # soft
    #def forward(self, data):
    #    attention = self.sig(torch.einsum('btd,kd->btk', data, self.memory_block) / (self.dim**0.5))
    
        # Softmax weighting – differentiable, gives gradient to all memory slots
    #    weights = torch.softmax(attention / (self.dim**0.5), dim=-1)
    #    temporal_att = (weights * attention).sum(dim=-1)        # fully differentiable
    #    augment = torch.einsum('btk,kd->btd', attention, self.memory_block)
    #    return temporal_att, augment
    
    def forward(self, data):
        # Raw attention with sigmoid (fully differentiable)
        attention = self.sig(torch.einsum('btd,kd->btk', data, self.memory_block) / (self.dim**0.5))

        # Top‑k temporal attention – used for cross‑suppression losses
        # This gives a sharp, interpretable signal and allows losses to drive the correct memory
        # attention to zero when misused.
        k = max(self.nums // 16 + 1, 1)
        topk_vals, _ = torch.topk(attention, k, dim=-1)
        temporal_att = topk_vals.mean(dim=-1)          # shape [B, T]

        # Augment uses the full attention → remains fully differentiable
        augment = torch.einsum('btk,kd->btd', attention, self.memory_block)
        return temporal_att, augment
if __name__ == "__main__":
    mu = Memory_Unit(10,512).cuda()
    data = torch.randn((3,20,512)).cuda()
    mu(data)