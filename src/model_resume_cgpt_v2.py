import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from memory_resume_cgpt import Memory_Unit
#from PATCHED_Memory_Unit import Memory_Unit
from translayer_resume_cgpt import Transformer
from utils import norm
import torch.nn.functional as F


class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, T, D]
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


class LocalAttention(nn.Module):
    """Lightweight depthwise-local attention implemented via grouped convolutions.
    Preserves shape: input [B, T, D] -> output [B, T, D]
    """
    def __init__(self, dim, kernel=9):
        super().__init__()
        self.conv = nn.Sequential(
            # depthwise (groups=dim) for lightweight local interaction
            nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, T, D] -> convert to [B, D, T] for conv1d
        return x + self.conv(x.transpose(1, 2)).transpose(1, 2)


class ADCLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # keep original head (sigmoid) to avoid changing score scaling; simple to swap to softmax if desired
        #self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim), nn.Sigmoid())
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
            )

    def forward(self, x):
        return self.mlp(x)


class WSAD(Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums

        # core modules
        self.embedding = Temporal(input_size, 512)

        # NEW: local attention prior to transformer
        self.localatt = LocalAttention(512)

        self.triplet = nn.TripletMarginLoss(margin=0.3)
        #self.triplet = nn.TripletMarginLoss(margin=1)
        self.cls_head = ADCLS_head(1024, 1)

        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)

        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.25)

        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        self.relu = nn.ReLU()

    #def _reparameterize(self, mu, logvar):
    #    std = torch.exp(logvar).sqrt()
    #    epsilon = torch.randn_like(std)
    #    return mu + epsilon * std
    
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1))
        return kl_loss

    # NEW: temporal smoothness loss to reduce frame jitter
    def temporal_smoothness(self, pred):
        # pred: [B, T] or [B, 1] handled outside
        if pred.dim() == 2 and pred.size(1) > 1:
            return (pred[:, 1:] - pred[:, :-1]).abs().mean()
        return torch.tensor(0.0, device=pred.device)

    def forward(self, x):
        # Accepts either [B, N, T, D] or [B, T, D]
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        # embedding + local attention + transformer
        x = self.embedding(x)
        x = self.localatt(x)   # NEW
        x = self.selfatt(x)
        
        # Feature magnitudes (L2 norm per snippet) for RTFM loss
        feat_magnitudes = torch.norm(x, p=2, dim=-1)   # shape [B*n, T]

        #if self.flag == "Train":
        if self.flag == "Train":
            # split into normal / abnormal parts
            N_x = x[: b * n // 2]
            A_x = x[b * n // 2 :]

            # ---------- Memory reads ----------
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)
            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)

            # ---------- UR‑DMU Triplet ----------
            # Anchor: normal video → normal memory top‑k snippets
            _, N_idx = torch.topk(N_att, t // 16 + 1, dim=-1)
            anchor = torch.gather(N_x, 1,
                                  N_idx.unsqueeze(2).expand(-1, -1, N_x.size(-1))
                                  ).mean(1).reshape(b // 2, n, -1).mean(1)

            # Positive: abnormal video → normal memory top‑k snippets (normal part)
            _, P_idx = torch.topk(N_Aatt, t // 16 + 1, dim=-1)
            positive = torch.gather(A_x, 1,
                                    P_idx.unsqueeze(2).expand(-1, -1, A_x.size(-1))
                                    ).mean(1).reshape(b // 2, n, -1).mean(1)

            # Negative: abnormal video → abnormal memory top‑k snippets (abnormal part)
            _, A_idx = torch.topk(A_att, t // 16 + 1, dim=-1)
            negative = torch.gather(A_x, 1,
                                    A_idx.unsqueeze(2).expand(-1, -1, A_x.size(-1))
                                    ).mean(1).reshape(b // 2, n, -1).mean(1)

            anchor   = norm(anchor)
            positive = norm(positive)
            negative = norm(negative)
            triplet_margin_loss = self.triplet(anchor, positive, negative)

            # ---------- VAE encoding ----------
            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            # Light stochasticity (20% noise)
            N_aug_new = 0.8 * N_aug_mu + 0.2 * self._reparameterize(N_aug_mu, N_aug_var)
            A_aug_new = self.encoder_mu(A_aug)
            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)

            # ---------- Distance loss (UR‑DMU L_dis) with SQUARED norms ----------
            k_dis = max(t // 16 + 1, 1)
            _, A_idx_dis = torch.topk(A_att, k_dis, dim=-1)
            _, N_idx_dis = torch.topk(N_att, k_dis, dim=-1)

            abnormal_topk = torch.gather(A_aug_new, 1,
                                         A_idx_dis.unsqueeze(2).expand(-1, -1, A_aug_new.size(-1)))
            normal_topk   = torch.gather(N_aug_new, 1,
                                         N_idx_dis.unsqueeze(2).expand(-1, -1, N_aug_new.size(-1)))
            abnormal_topk = abnormal_topk.mean(1).reshape(b // 2, n, -1).mean(1)
            normal_topk   = normal_topk.mean(1).reshape(b // 2, n, -1).mean(1)

            # Squared L2 norms (exactly as paper)
            distance_loss = torch.relu(
                100.0 - abnormal_topk.pow(2).sum(1) + normal_topk.pow(2).sum(1)
            ).mean()

            # ---------- Feature fusion ----------
            x = torch.cat((x, (torch.cat([N_aug_new + A_Naug,
                                          A_aug_new + N_Aaug], dim=0))), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            # ---------- Optional: keep or remove feat_mag_loss ----------
            # If you want to keep the RTFM‑style feature magnitude loss (not in paper)
            # with weight 0.0 it is effectively disabled.
            feat_magnitudes = torch.norm(x, p=2, dim=-1)  # raw backbone features
            k_mag = max(t // 16 + 1, 1)
            N_mag = feat_magnitudes[: b * n // 2]
            A_mag = feat_magnitudes[b * n // 2 :]
            topk_A_mag = torch.topk(A_mag, k_mag, dim=-1)[0].mean(-1).mean()
            topk_N_mag = torch.topk(N_mag, k_mag, dim=-1)[0].mean(-1).mean()
            feat_mag_loss = torch.relu(100.0 - (topk_A_mag - topk_N_mag))
            mag_reg = 0.001 * (topk_A_mag ** 2 + topk_N_mag ** 2)
            feat_mag_loss = feat_mag_loss + mag_reg

            # Temporal smoothness (disabled)
            smooth_loss = self.temporal_smoothness(pre_att)

            return {
                "frame": pre_att,
                "triplet_margin": triplet_margin_loss,
                "kl_loss": kl_loss,
                "distance": distance_loss,
                "smooth_loss": smooth_loss,
                "feat_mag_loss": feat_mag_loss,   # kept but weight 0 in loss
                "A_att": A_att.reshape((b // 2, n, -1)).mean(1),
                "N_att": N_att.reshape((b // 2, n, -1)).mean(1),
                "A_Natt": A_Natt.reshape((b // 2, n, -1)).mean(1),
                "N_Aatt": N_Aatt.reshape((b // 2, n, -1)).mean(1),
            }
        
        else:
            # Inference path: keep second read to reinforce test-time augmentations
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)

            #_, A_aug2 = self.Amemory(A_aug)
            #_, N_aug2 = self.Nmemory(N_aug)
            #A_aug = A_aug + 0.1 * A_aug2
            #N_aug = N_aug + 0.1 * N_aug2

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)
            #mem_aug = 0.5 * (A_aug + N_aug)
            #x = torch.cat([x, mem_aug], dim=-1)

            #pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            #pre_att = self.cls_head(x) / 0.7
            pre_att = self.cls_head(x)
            pre_att = pre_att.reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}


if __name__ == "__main__":
    # quick smoke-test
    m = WSAD(input_size=1024, flag="Train", a_nums=60, n_nums=60).cuda()
    src = torch.rand(100, 32, 1024).cuda()
    out = m(src)["frame"]
    print(out.size())
