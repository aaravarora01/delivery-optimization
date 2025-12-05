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

        # standard transformer decoding

        T = tgt.size(1)

        tgt_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()

        D = self.decoder(tgt, H, tgt_mask=tgt_mask)  # (B,T,D)
        
        # More aggressive clipping for numerical stability
        # Use smaller range to prevent extreme values that can cause NaN
        D = torch.clamp(D, min=-100.0, max=100.0)
        
        # Replace any NaN/inf with zeros as fallback
        D = torch.where(torch.isfinite(D), D, torch.zeros_like(D))

        q = self.out_proj(D[:, -1])  # (B,D) last step
        
        # Replace any NaN/inf in query with zeros
        q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
        
        # Normalize query to prevent extreme values
        q_norm = torch.norm(q, dim=-1, keepdim=True) + 1e-8
        q = q / q_norm

        keys = self.ptr_proj(H)      # (B,N,D)
        
        # Replace any NaN/inf in keys with zeros
        keys = torch.where(torch.isfinite(keys), keys, torch.zeros_like(keys))
        
        # Normalize keys to prevent extreme values
        keys_norm = torch.norm(keys, dim=-1, keepdim=True) + 1e-8
        keys = keys / keys_norm

        # Einsum with epsilon in sqrt to prevent division issues
        scale = math.sqrt(keys.size(-1)) + 1e-8
        logits = torch.einsum("bd,bnd->bn", q, keys) / scale  # pointer scores
        
        # Clamp logits immediately after einsum
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        # Replace any NaN/inf in logits with zeros
        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))

        if self.edge_bias is not None and edge_feats is not None:

            # Replace any NaN/inf in edge_feats with zeros (defensive check)
            edge_feats = torch.where(torch.isfinite(edge_feats), edge_feats, torch.zeros_like(edge_feats))

            # Use last chosen index to index edge_feats for bias against candidates

            # We approximate with a pooled bias: bias_i = mean_j(bias_ij) to stay cheap

            bias = self.edge_bias(edge_feats)  # (B,N,N)
            
            # Replace any NaN/inf in bias with zeros
            bias = torch.where(torch.isfinite(bias), bias, torch.zeros_like(bias))

            bias = bias.mean(dim=1)            # (B,N)
            
            # Clamp bias after mean to prevent extreme values
            bias = torch.clamp(bias, min=-10.0, max=10.0)
            
            # Replace any NaN/inf after mean with zeros
            bias = torch.where(torch.isfinite(bias), bias, torch.zeros_like(bias))

            logits = logits + bias
            
            # Replace any NaN/inf in logits after adding bias
            logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))

        logits = logits.masked_fill(mask_visited, float("-inf"))

        # Check if all logits are masked (all -inf) - this would cause NaN in softmax/log_softmax
        # This shouldn't happen in normal operation, but handle gracefully to prevent NaN
        all_masked = mask_visited.all(dim=1)  # (B,) - True if all nodes visited for this batch
        if all_masked.any():
            # If all nodes are visited, we shouldn't be decoding. Set all logits to 0 as fallback
            # This prevents NaN but indicates a logic error upstream
            logits = logits.masked_fill(all_masked.unsqueeze(1), 0.0)

        return logits

    def forward_teacher_forced(self, x, target_idx, edge_feats=None):

        """

        x: (B,N,d_in); target_idx: (B,N) indices 0..N-1 of the true sequence

        Returns CE loss (teacher forcing).

        """

        B,N,_ = x.shape

        device = x.device

        # Check inputs for NaN/inf and replace with zeros
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        if edge_feats is not None:
            edge_feats = torch.where(torch.isfinite(edge_feats), edge_feats, torch.zeros_like(edge_feats))

        H = self.encode(x)
        # Replace any NaN/inf in encoder output
        H = torch.where(torch.isfinite(H), H, torch.zeros_like(H))

        # Start token

        tgt = self.query_start.expand(B,1,-1)

        loss = 0.0

        mask_visited = torch.zeros(B, N, dtype=torch.bool, device=device)

        for t in range(N):

            logits = self.decode_step(H, tgt, mask_visited, edge_feats=edge_feats)
            
            # Replace any NaN/inf in logits with zeros (defensive check)
            if not torch.isfinite(logits).all():
                import warnings
                warnings.warn(f"NaN/inf detected in logits at step {t}, replacing with zeros")
                logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))

            y = target_idx[:, t]  # (B,)

            step_loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            
            # Replace any NaN/inf in loss with zero (defensive check)
            if not torch.isfinite(step_loss):
                import warnings
                warnings.warn(f"NaN/inf detected in loss at step {t}, replacing with zero")
                step_loss = torch.tensor(0.0, device=step_loss.device, dtype=step_loss.dtype)
            
            loss += step_loss

            # append teacher token embedding: take the chosen node embedding as next query

            chosen = y  # teacher forcing

            mask_visited = mask_visited.scatter(1, chosen.unsqueeze(1), True)

            next_embed = H[torch.arange(B, device=device), chosen]  # (B,D)
            
            # Normalize embedding to prevent extreme values from propagating
            next_embed = self.embed_norm(next_embed)  # (B,D)

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

