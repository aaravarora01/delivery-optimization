# src/models/pointer_transformer.py

from dataclasses import dataclass

from typing import Optional, Tuple

import math, torch

import torch.nn as nn

import torch.nn.functional as F

def _positional_encoding(n, d, device):

    pe = torch.zeros(n, d, device=device)

    position = torch.arange(0, n, device=device).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))

    pe[:, 0::2] = torch.sin(position * div_term)

    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

class RelEdgeBias(nn.Module):

    """Tiny MLP that maps [dist, dtheta, same_street?] -> scalar bias for attention logits."""

    def __init__(self, in_dim=3, hidden=32):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(in_dim, hidden), nn.ReLU(),

            nn.Linear(hidden, 1)

        )

    def forward(self, edge_feats):  # (B, N, N, 3)

        B,N,_,_ = edge_feats.shape

        out = self.net(edge_feats).view(B, N, N)  # bias_ij
        
        # Clamp bias output to reasonable range to prevent extreme values
        out = torch.clamp(out, min=-10.0, max=10.0)
        
        # Replace any NaN/inf with zeros as fallback
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))

        return out

@dataclass

class PTConfig:

    d_model: int = 128

    nhead: int = 4

    d_ff: int = 256

    nlayers_enc: int = 2

    nlayers_dec: int = 2

    dropout: float = 0.1

    use_edge_bias: bool = True

class PointerTransformer(nn.Module):

    def __init__(self, d_in: int, cfg: PTConfig = PTConfig()):

        super().__init__()

        self.cfg = cfg

        self.inp = nn.Linear(d_in, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(

            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.d_ff, dropout=cfg.dropout, batch_first=True)

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.nlayers_enc)

        dec_layer = nn.TransformerDecoderLayer(

            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.d_ff, dropout=cfg.dropout, batch_first=True)

        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.nlayers_dec)

        self.query_start = nn.Parameter(torch.randn(1,1,cfg.d_model))

        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.ptr_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.edge_bias = RelEdgeBias() if cfg.use_edge_bias else None
        
        # LayerNorm for stabilizing embeddings
        self.embed_norm = nn.LayerNorm(cfg.d_model)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with smaller std values for numerical stability."""
        # Initialize query_start with smaller std
        nn.init.normal_(self.query_start, mean=0.0, std=0.01)
        
        # Initialize linear layers with xavier uniform
        for module in [self.inp, self.out_proj, self.ptr_proj]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Initialize edge bias network with smaller weights
        if self.edge_bias is not None:
            for m in self.edge_bias.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def encode(self, x):

        # x: (B,N,d_in) node features

        h = self.inp(x)

        h = h + _positional_encoding(h.size(1), h.size(2), h.device)

        H = self.encoder(h)  # (B,N,D)

        return H

    def decode_step(self, H, tgt, mask_visited, edge_feats=None, last_node_idx=None):

        """

        H: (B,N,D) encoder outputs

        tgt: (B,t,D) decoded tokens so far (queries)

        mask_visited: (B,N) bool mask for nodes already chosen

        edge_feats: (B,N,N,3) for optional attention bias

        Returns logits over N nodes for next pick.

        """

        T = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()

        D = self.decoder(tgt, H, tgt_mask=tgt_mask)  # (B,T,D)
        D = torch.clamp(D, min=-100.0, max=100.0)

        q = self.out_proj(D[:, -1])  # (B,D) last step
        q = torch.clamp(q, min=-10.0, max=10.0)

        keys = self.ptr_proj(H)      # (B,N,D)
        keys = torch.clamp(keys, min=-10.0, max=10.0)

        scale = math.sqrt(keys.size(-1)) + 1e-8
        logits = torch.einsum("bd,bnd->bn", q, keys) / scale  # pointer scores

        if self.edge_bias is not None and edge_feats is not None:
            bias = self.edge_bias(edge_feats)  # (B,N,N)
            bias = bias.mean(dim=1)            # (B,N)
            bias = torch.clamp(bias, min=-2.0, max=2.0)
            logits = logits + 0.1 * bias

        logits = logits.masked_fill(mask_visited, float(-1e9))

        all_masked = mask_visited.all(dim=1)
        if all_masked.any():
            logits = logits.masked_fill(all_masked.unsqueeze(1), 0.0)

        return logits

    def forward_teacher_forced(self, x, target_idx, edge_feats=None):

        """

        x: (B,N,d_in); target_idx: (B,N) indices 0..N-1 of the true sequence

        Returns CE loss (teacher forcing).

        """

        B,N,_ = x.shape

        device = x.device

        H = self.encode(x)
        tgt = self.query_start.expand(B,1,-1)
        loss = 0.0
        mask_visited = torch.zeros(B, N, dtype=torch.bool, device=device)

        for t in range(N):
            logits = self.decode_step(H, tgt, mask_visited, edge_feats=edge_feats)
            y = target_idx[:, t]  # (B,)
            step_loss = F.cross_entropy(logits, y, reduction='mean')
            loss += step_loss

            chosen = y  # teacher forcing
            mask_visited = mask_visited.scatter(1, chosen.unsqueeze(1), True)
            next_embed = H[torch.arange(B, device=device), chosen]  # (B,D)
            next_embed = self.embed_norm(next_embed)
            tgt = torch.cat([tgt, next_embed.unsqueeze(1)], dim=1)

        return loss / N

    @torch.no_grad()

    def greedy_decode(self, x, edge_feats=None):

        B,N,_ = x.shape

        device = x.device

        H = self.encode(x)

        tgt = self.query_start.expand(B,1,-1)

        mask_visited = torch.zeros(B, N, dtype=torch.bool, device=device)

        seq = []

        for _ in range(N):

            # Check if all nodes are already visited (shouldn't happen, but prevent infinite loop)
            if mask_visited.all():
                break

            logits = self.decode_step(H, tgt, mask_visited, edge_feats=edge_feats)

            choice = torch.argmax(logits, dim=1)  # (B,)

            seq.append(choice)

            mask_visited = mask_visited.scatter(1, choice.unsqueeze(1), True)

            next_embed = H[torch.arange(B, device=device), choice]
            
            # Normalize embedding to prevent extreme values from propagating
            next_embed = self.embed_norm(next_embed)  # (B,D)

            tgt = torch.cat([tgt, next_embed.unsqueeze(1)], dim=1)

        seq = torch.stack(seq, dim=1)  # (B,N)

        return seq

