import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from memory import Memory_Unit
from translayer import Transformer
from utils import norm


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
        self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim), nn.Sigmoid())

    def forward(self, x):
        return self.mlp(x)



class WSAD(Module):
    def __init__(self, input_size, flag, a_nums, n_nums, config=None):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums
        self.config = config if config is not None else {}

        # core embedding
        self.embedding = Temporal(input_size, 512)

        # optional modules
        self.use_local_att = self.config.get("use_local_att", True)
        print(f'use_local_att is {self.use_local_att}')
        self.localatt = LocalAttention(512) if self.use_local_att else nn.Identity()

        self.use_transformer = self.config.get("use_transformer", True)
        print(f'use_transformer is {self.use_transformer}')
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.5) if self.use_transformer else nn.Identity()

        # memory units
        self.use_dual_memory = self.config.get("use_dual_memory", True)
        print(f'use_dual_memory is {self.use_dual_memory}')
        self.Amemory = Memory_Unit(nums=a_nums, dim=512) if self.use_dual_memory else nn.Identity()
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512) if self.use_dual_memory else nn.Identity()

        # VAE / latent encoding
        self.use_vae = self.config.get("use_vae", True)
        print(f'use_vae is {self.use_vae}')
        self.encoder_mu = nn.Sequential(nn.Linear(512, 512)) if self.use_vae else nn.Identity()
        self.encoder_var = nn.Sequential(nn.Linear(512, 512)) if self.use_vae else nn.Identity()
        self.relu = nn.ReLU()

        # classifier
        #self.cls_head = ADCLS_head(1024, 1)
        cls_in_dim = 512
        if self.use_dual_memory:
            cls_in_dim += 512  # memory augmentation
        self.cls_head = ADCLS_head(cls_in_dim, 1)
        self.triplet = nn.TripletMarginLoss(margin=1)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        if not self.use_vae:
            return torch.tensor(0.0, device=mu.device)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1))
        return kl_loss

    def temporal_smoothness(self, pred):
        if pred.dim() == 2 and pred.size(1) > 1:
            return (pred[:, 1:] - pred[:, :-1]).abs().mean()
        return torch.tensor(0.0, device=pred.device)

    def forward(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        # embedding
        x = self.embedding(x)

        # optional modules
        x = self.localatt(x)
        x = self.selfatt(x)

        # placeholders for outputs
        A_att = N_att = A_Natt = N_Aatt = distance = torch.tensor(0.0, device=x.device)
        triplet_margin_loss = kl_loss = smooth_loss = torch.tensor(0.0, device=x.device)

        if self.flag == "Train" and self.use_dual_memory:
            N_x = x[: b * n // 2]
            A_x = x[b * n // 2 :]

            # first memory read
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)
            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)

            # second memory read for augmentation
            _, A_aug2 = self.Amemory(A_aug)
            _, N_aug2 = self.Nmemory(N_aug)
            A_aug = A_aug + 0.5 * A_aug2
            N_aug = N_aug + 0.5 * N_aug2

            # triplet margin loss
            _, A_index = torch.topk(A_att, t // 16 + 1, dim=-1)
            negative_ax = torch.gather(A_x, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            _, N_index = torch.topk(N_att, t // 16 + 1, dim=-1)
            anchor_nx = torch.gather(N_x, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            _, P_index = torch.topk(N_Aatt, t // 16 + 1, dim=-1)
            positivte_nx = torch.gather(A_x, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            triplet_margin_loss = self.triplet(norm(anchor_nx), norm(positivte_nx), norm(negative_ax))

            # VAE latent encoding
            if self.use_vae:
                N_aug_mu = self.encoder_mu(N_aug)
                N_aug_var = self.encoder_var(N_aug)
                N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)
            else:
                N_aug_new = N_aug
                N_aug_mu, N_aug_var = N_aug, N_aug

            # classifier input
            x_cat = torch.cat([N_aug_new + A_Naug, A_aug + N_Aaug], dim=0)
            x_final = torch.cat([x, x_cat], dim=-1)
            pre_att = self.cls_head(x_final).reshape((b, n, -1)).mean(1)

            # temporal smoothness
            smooth_loss = self.temporal_smoothness(pre_att)

            # latent loss
            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

            # distance (same as original)
            negative_ax_new = torch.gather(self.encoder_mu(A_aug) if self.use_vae else A_aug, 1,
                                           A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)
            anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)
            distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()

            return {
                "frame": pre_att,
                "triplet_margin": triplet_margin_loss,
                "kl_loss": kl_loss,
                "distance": distance,
                "smooth_loss": smooth_loss,
                "A_att": A_att.reshape((b // 2, n, -1)).mean(1),
                "N_att": N_att.reshape((b // 2, n, -1)).mean(1),
                "A_Natt": A_Natt.reshape((b // 2, n, -1)).mean(1),
                "N_Aatt": N_Aatt.reshape((b // 2, n, -1)).mean(1),
            }

        else:
            # inference / optional dual memory
            if self.use_dual_memory:
                _, A_aug = self.Amemory(x)
                _, N_aug = self.Nmemory(x)
                _, A_aug2 = self.Amemory(A_aug)
                _, N_aug2 = self.Nmemory(N_aug)
                A_aug = A_aug + 0.5 * A_aug2
                N_aug = N_aug + 0.5 * N_aug2
                if self.use_vae:
                    A_aug = self.encoder_mu(A_aug)
                    N_aug = self.encoder_mu(N_aug)
                x = torch.cat([x, A_aug + N_aug], dim=-1)

            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            return {
                "frame": pre_att,
                "triplet_margin": triplet_margin_loss,
                "kl_loss": kl_loss,
                "distance": distance,
                "smooth_loss": smooth_loss,
                "A_att": A_att,
                "N_att": N_att,
                "A_Natt": A_Natt,
                "N_Aatt": N_Aatt,
            }
