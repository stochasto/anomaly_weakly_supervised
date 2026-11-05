
class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label, step):
        loss = {}
        _label = _label.float()

        triplet = result.get("triplet_margin", torch.tensor(0.0, device=_label.device))
        
        # orig att
        #att = result['frame']
        
        # new att
        att_raw = result['frame']
        att = 0.7 * att_raw.detach() + 0.3 * att_raw
        # optional: clamp to avoid extreme logits (VERY helpful for stability)
        att = torch.clamp(att, 1e-4, 1 - 1e-4)
        
        if ((step + 1) % 5 == 0):
            print("Att raw mean:", att_raw.mean().item())
            print("Att stabilized mean:", att.mean().item())
            print("Att std:", att.std().item())
        
        A_att = result.get("A_att", torch.zeros_like(att))
        N_att = result.get("N_att", torch.zeros_like(att))
        A_Natt = result.get("A_Natt", torch.zeros_like(att))
        N_Aatt = result.get("N_Aatt", torch.zeros_like(att))
        kl_loss = result.get("kl_loss", torch.tensor(0.0, device=_label.device))
        distance = result.get("distance", torch.tensor(0.0, device=_label.device))

         # ✅ NEW: smooth loss
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))

        b = max(_label.size(0) // 2, 1)
        t = max(att.size(1), 1)

        #orig
        anomaly_topk = max(t // 16 + 1, 1)
        #anomaly_topk = max(t // 32 + 1, 1)
        #anomaly_topk = max(t // 64, 1)

        anomaly = torch.topk(att, anomaly_topk, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)
        # ===== CURRICULUM WEIGHT FOR ATTENTION LOSS =====
        #att_weight = min(0.1, 0.01 + 0.001 * step)
        
        ## margin based
        # Compute means
        anomaly_mean = att.mean()
        topk_mean = torch.topk(att, anomaly_topk, dim=-1)[0].mean()
        # Margin separation (CRITICAL)
        margin = 0.2
        separation_loss = torch.relu(margin - (topk_mean - anomaly_mean))

        panomaly = torch.topk(1 - N_Aatt, anomaly_topk, dim=-1)[0].mean(-1)
        panomaly = panomaly[:b]
        panomaly_loss = self.bce(panomaly, torch.ones_like(panomaly))

        A_att_topk = torch.topk(A_att, anomaly_topk, dim=-1)[0].mean(-1)
        A_att_topk = A_att_topk[:b]
        A_loss = self.bce(A_att_topk, torch.ones_like(A_att_topk))

        N_loss = self.bce(N_att, torch.ones_like(N_att))
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt))
        
        # sep loss
        normal_mask = (att < att.mean(dim=1, keepdim=True)).float()
        abn_mask = (att >= att.mean(dim=1, keepdim=True)).float()
        sep_loss = (att * normal_mask).mean() - (att * abn_mask).mean()
        
        # sparsit loss
        sparsity_loss = att.mean()
        
        if ((step + 1) % 5 == 0):
            print("Anomaly mean:", att.mean().item())
            print("Top-k mean:", torch.topk(att, anomaly_topk).values.mean().item())

        # stronger triplet/kl scaling to encourage sharper localization
        mult_dual = 0.15
        mult_triplet = 0.3
        mult_kl = 0.005
        mult_dist = 0.0001
        mult_smooth = 0.05
        mult_sep = 0.2
        mult_sparsity = 0.1
        mult_margin_sep = 0.5
        att_weight = 0.05
        total_loss = (
            att_weight*anomaly_loss
            + mult_dual * (A_loss + panomaly_loss + N_loss + A_Nloss)
            + mult_triplet * triplet
            + mult_kl * kl_loss
            + mult_dist * distance
            + mult_smooth * smooth_loss
            + mult_sep * sep_loss
            + mult_sparsity * sparsity_loss
            + mult_margin_sep * separation_loss
        )

        # gave 83.6% with num_seg 300, anum = nnum = 300
        #mult_dual = 0.2
        #mult_triplet = 0.2
        #mult_kl = 0.01
        #mult_dist = 0.0001
        #mult_smooth = 0.01
        #total_loss = (
        #    anomaly_loss
        #    + mult_dual * (A_loss + panomaly_loss + N_loss + A_Nloss)
        #    + mult_triplet * triplet
        #    + mult_kl * kl_loss
        #    + mult_dist * distance
        #    + mult_smooth * smooth_loss
        #)

        # orig loss
        #total_loss = (
        #    anomaly_loss
        #    + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss)
        #    + 0.2 * triplet
        #    + 0.005 * kl_loss
        #    + 0.0001 * distance
        #)

        loss["total_loss"] = total_loss
        loss["att_loss"] = anomaly_loss
        loss["N_Aatt"] = panomaly_loss
        loss["A_loss"] = A_loss
        loss["N_loss"] = N_loss
        loss["A_Nloss"] = A_Nloss
        loss["triplet"] = triplet
        loss["kl_loss"] = kl_loss
        loss["smooth_loss"] = smooth_loss 

        return total_loss, loss