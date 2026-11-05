import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class Memory_Unit(Module):
    """
    Attention-based memory unit with:
      - Multi-head attention over prototypes (replaces single-head / Top-K readout)
      - Learnable temperature for attention sharpening
      - Attention-based prototype refinement (in-forward, differentiable)
    """

    def __init__(self, nums, dim, temperature=2.0, topk=4, n_heads=4):
        super().__init__()
        self.dim = dim
        self.nums = nums
        self.topk = topk                      # kept for backwards compat, unused
        self.n_heads = n_heads
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.head_dim = dim // n_heads

        # Learnable temperature (initialised at the previous hard-coded 2.0)
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

        # Per-head query and key projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)

        # Learnable scalar gate for prototype refinement (sigmoid(-2) ≈ 0.12)
        # Small initial value → the network starts close to the original memory
        # and can learn to trust the attention-based refinement as training goes.
        self.update_alpha = nn.Parameter(torch.tensor(-2.0))

        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.memory_block)

    def forward(self, data, external_memory=None):
        """
        Args:
            data:            [B, T, D]
            external_memory: optional [K, D] (base + PEN-augmented prototypes)
        Returns:
            temporal_att:    [B, T]
            augment:         [B, T, D]
        """
        B, T, D = data.shape
        data_n = F.normalize(data, p=2, dim=-1)

        # ---- base memory ----
        if external_memory is not None:
            base_memory = F.normalize(external_memory, p=2, dim=-1)
        else:
            base_memory = F.normalize(self.memory_block, p=2, dim=-1)
        K = base_memory.shape[0]

        # ---- multi-head projections ----
        q = self.q_proj(data_n)                                       # [B, T, D]
        k = self.k_proj(base_memory)                                  # [K, D]

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, T, hd]
        k = k.view(K, self.n_heads, self.head_dim).permute(1, 0, 2)   # [H, K, hd]

        q_n = F.normalize(q, p=2, dim=-1)
        k_n = F.normalize(k, p=2, dim=-1)

        # Per-head cosine similarity
        sim = torch.einsum('bhtd,hkd->bhtk', q_n, k_n)                # [B, H, T, K]

        # ---- learnable temperature ----
        temp = torch.clamp(self.temperature, min=0.1)

        raw_att = torch.sigmoid(sim * temp)                           # [B, H, T, K]
        weights = torch.softmax(raw_att * temp, dim=-1)               # [B, H, T, K]

        # ---- temporal attention (head-averaged) ----
        temporal_att = (weights * raw_att).sum(dim=-1).mean(dim=1)    # [B, T]

        # ---- average attention across heads for augmentation ----
        attn_avg = raw_att.mean(dim=1)                                # [B, T, K]

        # ---- attention-based prototype refinement ----
        # Each prototype is pulled toward the attention-weighted mean of the
        # snippets it responded to. The gate alpha is learnable, and the
        # refined memory is used only for this forward pass — the base
        # prototypes remain learned via gradient descent.
        attn_sum = attn_avg.sum(dim=(0, 1)).unsqueeze(-1) + 1e-6      # [K, 1]
        weighted_feats = torch.einsum('btk,btd->kd', attn_avg, data_n)  # [K, D]
        context = weighted_feats / attn_sum                            # [K, D]
        context = F.normalize(context, p=2, dim=-1)

        alpha = torch.sigmoid(self.update_alpha)
        refined_memory = (1.0 - alpha) * base_memory + alpha * context  # [K, D]

        # ---- augmentation with refined memory ----
        augment = torch.einsum('btk,kd->btd', attn_avg, refined_memory) # [B, T, D]

        return temporal_att, augment


if __name__ == "__main__":
    mu = Memory_Unit(nums=200, dim=512).cuda()
    data = torch.randn((3, 20, 512)).cuda()
    temporal_att, augment = mu(data)
    print("temporal_att:", temporal_att.shape)   # [3, 20]
    print("augment:", augment.shape)             # [3, 20, 512]