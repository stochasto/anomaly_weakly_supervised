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
        self.topk = topk
        
        # Learnable temperature initialization
        self.temp = nn.Parameter(torch.tensor(temperature))
        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.memory_block)

    def forward(self, data):
        # [B, T, D] -> Normalize
        data = F.normalize(data, p=2, dim=-1)
        memory = F.normalize(self.memory_block, p=2, dim=-1)

        # 1. Cosine similarity: C (Values between -1 and 1)
        logits = torch.einsum('btd,kd->btk', data, memory)   # [B, T, K]

        # 2. Scale by temperature (allow it to learn sharpness directly on logits)
        temp = torch.clamp(self.temp, min=1.0, max=10.0)
        scaled_logits = logits * temp

        # 3. FIX: Apply Softmax DIRECTLY to the scaled logits, NOT a sigmoid
        # This yields W, a sharp distribution over memory slots
        attention_W = torch.softmax(scaled_logits, dim=-1)   

        # 4. Temporal Attention Map
        k = min(self.topk, self.nums)
        #topk_vals, _ = torch.topk(attention_W, k, dim=-1)
        #temporal_att = topk_vals.mean(dim=-1)                # [B, T]
        
        # Better: Use the top-k raw similarities for the temporal score
        topk_logits, _ = torch.topk(logits, k, dim=-1)
        temporal_att = torch.sigmoid(topk_logits.mean(dim=-1)) # Maps to [0, 1]

        # 5. Augmentation: Multiply sharp attention by memory prototypes
        augment = torch.einsum('btk,kd->btd', attention_W, memory)

        return temporal_att, augment