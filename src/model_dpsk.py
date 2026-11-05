import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import math

# Enhanced Memory Unit
class EnhancedMemory_Unit(Module):
    def __init__(self, nums, dim, temperature=0.1):
        super(EnhancedMemory_Unit, self).__init__()
        self.nums = nums
        self.dim = dim
        self.temperature = temperature
        
        # Initialize memory with Xavier initialization
        self.memory = nn.Parameter(torch.randn(nums, dim))
        nn.init.xavier_uniform_(self.memory)
        
        self.attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        b, t, d = x.shape
        
        # Cross-attention between input and memory
        memory_expanded = self.memory.unsqueeze(0).expand(b, -1, -1)
        att_output, att_weights = self.attention(
            x, memory_expanded, memory_expanded
        )
        
        # Temperature-scaled similarity
        similarity = torch.matmul(x, self.memory.T) / self.temperature
        att = torch.softmax(similarity, dim=-1)
        
        # Memory augmentation
        aug = torch.matmul(att, self.memory)
        
        return att, aug + self.dropout(att_output)

# Enhanced Temporal Module
class ImprovedTemporal(Module):
    def __init__(self, input_size, out_size):
        super(ImprovedTemporal, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
        )
        
    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)
        return x

# Multi-Scale Transformer
class MultiScaleTransformer(Module):
    def __init__(self, dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()
        
        # Basic transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers)
            for _ in range(3)  # Multi-scale processing
        ])
        
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        
    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.transformer_layers):
            out = layer(x)
            out = self.layer_norms[i](out)
            outputs.append(out)
        
        # Learnable weighted combination
        weights = torch.softmax(self.attention_weights, dim=0)
        weighted_output = sum(w * out for w, out in zip(weights, outputs))
        return weighted_output

# Improved Classification Head
class ImprovedADCLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        if len(x.shape) == 3:
            # Flatten temporal dimension for MLP
            b, t, d = x.shape
            x = x.reshape(b * t, d)
            x = self.mlp(x)
            x = x.reshape(b, t, -1)
        else:
            x = self.mlp(x)
        return x

# Main Improved WSAD Model
class WSAD(Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums

        # Enhanced components
        self.embedding = ImprovedTemporal(input_size, 512)
        self.multi_scale_transformer = MultiScaleTransformer(512, 8, 2, 512)
        self.triplet = nn.TripletMarginLoss(margin=1.0, p=2)
        self.cls_head = ImprovedADCLS_head(1024, 1)
        
        # Enhanced memory units
        self.Amemory = EnhancedMemory_Unit(nums=a_nums, dim=512)
        self.Nmemory = EnhancedMemory_Unit(nums=n_nums, dim=512)
        
        # Additional components
        self.feature_refinement = nn.Linear(512, 512)
        self.attention_pool = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # Variational autoencoder components
        self.encoder_mu = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.encoder_var = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Loss components
        self.contrastive_loss = nn.CosineEmbeddingLoss()
        self.relu = nn.ReLU()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def _extract_salient_features(self, x, indices, b, n):
        """Extract and pool salient features with proper dimension handling"""
        # Reshape indices to match x dimensions
        if len(indices.shape) == 4:
            # indices shape: [batch, num_crops, temporal, topk]
            batch_size, num_crops, temporal, topk = indices.shape
            indices = indices.reshape(batch_size * num_crops, temporal, topk)
        
        # Ensure x and indices have compatible shapes
        if len(x.shape) == 3 and len(indices.shape) == 3:
            # x shape: [batch*crops, temporal, features]
            # indices shape: [batch*crops, temporal, topk]
            salient_features = torch.gather(
                x, 
                1, 
                indices.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))
            )
            
            # Multi-scale pooling
            avg_pool = salient_features.mean(2)  # Average over topk dimension
            max_pool = salient_features.max(2)[0]  # Max over topk dimension
            
            # Attention-based pooling
            salient_flat = salient_features.reshape(salient_features.size(0), -1, salient_features.size(-1))
            attn_output, _ = self.attention_pool(salient_flat, salient_flat, salient_flat)
            attn_pool = attn_output.mean(1)
            
            combined = (avg_pool.mean(1) + max_pool.mean(1) + attn_pool) / 3.0
            
            # Reshape back to original batch structure
            if n > 1:
                combined = combined.reshape(b//2, n, -1).mean(1)
            return combined
        else:
            # Fallback to simple mean pooling if dimensions don't match
            return x.mean(1)

    def forward(self, x):
        # Store original shape for reconstruction
        original_shape = x.shape
        
        # Handle different input shapes
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
            
        # Enhanced feature processing
        x = self.embedding(x)
        x = self.multi_scale_transformer(x)
        x = self.feature_refinement(x) + x  # Residual connection
        
        if self.flag == "Train":
            # Split normal and abnormal segments
            split_point = b * n // 2
            N_x = x[:split_point]  # Normal part
            A_x = x[split_point:]  # Abnormal part
            
            # Enhanced memory operations
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)
            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)
            
            # Improved feature selection with adaptive top-k
            topk_size = max(t // 16 + 1, 4)  # Adaptive top-k size
            
            # Extract topk indices with proper dimension handling
            _, A_index = torch.topk(A_att, topk_size, dim=-1)
            _, N_index = torch.topk(N_att, topk_size, dim=-1) 
            _, P_index = torch.topk(N_Aatt, topk_size, dim=-1)
            
            # Extract salient features
            negative_ax = self._extract_salient_features(A_x, A_index, b, n)
            anchor_nx = self._extract_salient_features(N_x, N_index, b, n)
            positive_nx = self._extract_salient_features(A_x, P_index, b, n)
            
            # Enhanced loss computation
            triplet_loss = self.triplet(self.norm(anchor_nx), self.norm(positive_nx), self.norm(negative_ax))
            
            # Additional contrastive loss
            contrastive_label = torch.ones(anchor_nx.size(0)).to(anchor_nx.device)
            contrastive_loss = self.contrastive_loss(anchor_nx, positive_nx, contrastive_label)
            
            # Variational encoding with proper reshaping
            N_aug_flat = N_aug.reshape(-1, N_aug.size(-1))
            N_aug_mu = self.encoder_mu(N_aug_flat)
            N_aug_logvar = self.encoder_var(N_aug_flat)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_logvar).reshape_as(N_aug)
            
            A_aug_flat = A_aug.reshape(-1, A_aug.size(-1))
            A_aug_mu = self.encoder_mu(A_aug_flat)
            A_aug_logvar = self.encoder_var(A_aug_flat)
            A_aug_new = self._reparameterize(A_aug_mu, A_aug_logvar).reshape_as(A_aug)
            
            # KL divergence loss
            kl_loss = self.latent_loss(N_aug_mu, N_aug_logvar) + self.latent_loss(A_aug_mu, A_aug_logvar)
            
            # Process A_Naug and N_Aaug through encoders
            A_Naug_flat = A_Naug.reshape(-1, A_Naug.size(-1))
            A_Naug = self.encoder_mu(A_Naug_flat).reshape_as(A_Naug)
            
            N_Aaug_flat = N_Aaug.reshape(-1, N_Aaug.size(-1))
            N_Aaug = self.encoder_mu(N_Aaug_flat).reshape_as(N_Aaug)
            
            # Enhanced feature fusion
            memory_fusion = N_aug_new + A_Naug + A_aug_new + N_Aaug
            enhanced_features = torch.cat((x, memory_fusion), dim=-1)
            
            # Classification
            pre_att = self.cls_head(enhanced_features)
            
            # Reshape output to match original batch structure
            if n > 1:
                pre_att = pre_att.reshape(b, n, -1).mean(1)
            else:
                pre_att = pre_att.reshape(b, -1)
            
            # Distance regularization
            anchor_nx_new = self._extract_salient_features(N_aug_new, N_index, b, n)
            negative_ax_new = self._extract_salient_features(A_aug_new, A_index, b, n)
            distance = torch.relu(1.0 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()
            
            # Reshape attention outputs for return
            A_att_reshaped = A_att.reshape(b//2, n, -1).mean(1) if n > 1 else A_att.mean(1)
            N_att_reshaped = N_att.reshape(b//2, n, -1).mean(1) if n > 1 else N_att.mean(1)
            A_Natt_reshaped = A_Natt.reshape(b//2, n, -1).mean(1) if n > 1 else A_Natt.mean(1)
            N_Aatt_reshaped = N_Aatt.reshape(b//2, n, -1).mean(1) if n > 1 else N_Aatt.mean(1)
            
            return {
                "frame": pre_att,
                'triplet_margin': triplet_loss,
                'contrastive_loss': contrastive_loss,
                'kl_loss': kl_loss,
                'distance': distance,
                'total_loss': triplet_loss + 0.3 * contrastive_loss + 0.1 * kl_loss + 0.1 * distance,
                'A_att': A_att_reshaped,
                "N_att": N_att_reshaped,
                "A_Natt": A_Natt_reshaped,
                "N_Aatt": N_Aatt_reshaped
            }
        else:           
            # Enhanced inference
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)
            
            # Encode memory features
            A_aug_flat = A_aug.reshape(-1, A_aug.size(-1))
            A_aug = self.encoder_mu(A_aug_flat).reshape_as(A_aug)
            
            N_aug_flat = N_aug.reshape(-1, N_aug.size(-1))
            N_aug = self.encoder_mu(N_aug_flat).reshape_as(N_aug)
            
            enhanced_features = torch.cat([x, A_aug + N_aug], dim=-1)
            pre_att = self.cls_head(enhanced_features)
            
            # Reshape output to match original batch structure
            if n > 1:
                pre_att = pre_att.reshape(b, n, -1).mean(1)
            else:
                pre_att = pre_att.reshape(b, -1)
            
            return {"frame": pre_att}
    
    def norm(self, x):
        """Normalize features"""
        return torch.nn.functional.normalize(x, p=2, dim=-1)

# Utility functions
def get_optimizer(model, learning_rate=1e-4):
    """Get optimized optimizer with layer-wise learning rates"""
    return torch.optim.AdamW([
        {'params': model.embedding.parameters(), 'lr': learning_rate},
        {'params': model.multi_scale_transformer.parameters(), 'lr': learning_rate},
        {'params': model.cls_head.parameters(), 'lr': learning_rate * 2},
        {'params': model.Amemory.parameters(), 'lr': learning_rate * 2},
        {'params': model.Nmemory.parameters(), 'lr': learning_rate * 2},
        {'params': model.encoder_mu.parameters(), 'lr': learning_rate},
        {'params': model.encoder_var.parameters(), 'lr': learning_rate},
    ], weight_decay=1e-4, eps=1e-8)

def get_scheduler(optimizer, epochs):
    """Get cosine annealing scheduler with warm restarts"""
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=epochs//4, T_mult=2, eta_min=1e-6
    )

if __name__ == "__main__":
    # Test the improved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with different input shapes
    print("Testing with 4D input (b, n, t, d)...")
    m = ImprovedWSAD(input_size=1024, flag="Train", a_nums=60, n_nums=60).to(device)
    
    # Test with 4D input (batch, crops, temporal, features)
    src_4d = torch.rand(4, 10, 32, 1024).to(device)  # batch=4, crops=10, temporal=32, features=1024
    out_4d = m(src_4d)
    print("4D Input - Output size:", out_4d["frame"].size())
    
    print("\nTesting with 3D input (b, t, d)...")
    src_3d = torch.rand(8, 32, 1024).to(device)  # batch=8, temporal=32, features=1024
    out_3d = m(src_3d)
    print("3D Input - Output size:", out_3d["frame"].size())
    
    print("\nLoss components:")
    for key, value in out_4d.items():
        if 'loss' in key or 'distance' in key:
            print(f"{key}: {value.item():.4f}")