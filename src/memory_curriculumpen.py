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
        
    def forward(self, data, external_memory=None):
        data = F.normalize(data, p=2, dim=-1)
        if external_memory is not None:
            memory = F.normalize(external_memory, p=2, dim=-1)
        else:
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