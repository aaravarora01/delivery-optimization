# src/models/pointer_transformer.py

from dataclasses import dataclass
from typing import Optional, Tuple
import math, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

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

    def encode(self, x):

        # x: (B,N,d_in) node features

        h = self.inp(x)

        h = h + _positional_encoding(h.size(1), h.size(2), h.device)

        if self.training:
            H = checkpoint.checkpoint(self.encoder, h)
        else:
            H = self.encoder(h)

        return H

    def decode_step(self, H, tgt, mask_visited, edge_feats=None, last_node_idx=None):
        """
        H: (B,N,D) encoder outputs
        tgt: (B,t,D) decoded tokens so far (queries)
        mask_visited: (B,N) bool mask for nodes already chosen
        edge_feats: (B,N,N,3) for optional attention bias
        last_node_idx: (B,) indices of last selected node (for edge bias)
        Returns logits over N nodes for next pick.
        """
        B, N, D = H.shape
        T = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()
        
        if self.training:
            D_out = checkpoint.checkpoint(self.decoder, tgt, H, tgt_mask)
        else:
            D_out = self.decoder(tgt, H, tgt_mask=tgt_mask)
        
        q = self.out_proj(D_out[:, -1])  # (B, D) last step
        keys = self.ptr_proj(H)          # (B, N, D)

        logits = torch.einsum("bd,bnd->bn", q, keys) / math.sqrt(keys.size(-1))  # (B, N)
        
        # Debug: verify shape
        assert logits.shape == (B, N), f"Logits shape wrong: {logits.shape}, expected ({B}, {N})"

        if self.edge_bias is not None and edge_feats is not None and last_node_idx is not None:
            # Get edge features FROM last selected node TO all nodes
            bias_full = self.edge_bias(edge_feats)  # (B, N, N)
            # Extract edges from last_node_idx to all other nodes
            bias = bias_full[torch.arange(B, device=H.device), last_node_idx, :]  # (B, N)
            bias = torch.clamp(bias, min=-10.0, max=10.0)
            logits = logits + bias
            
            # Debug: verify shape after bias
            assert logits.shape == (B, N), f"Logits shape wrong after bias: {logits.shape}, expected ({B}, {N})"

        # Clamp logits before masking
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        logits = logits.masked_fill(mask_visited, -1e4)

        return logits  # Must be (B, N)

    def forward_teacher_forced(self, x, target_idx, edge_feats=None):
        """
        x: (B,N,d_in); target_idx: (B,N) indices 0..N-1 of the true sequence
        Returns CE loss (teacher forcing).
        """
        B, N, _ = x.shape
        device = x.device

        # Check input features
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Invalid input x detected")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        H = self.encode(x)
        
        # Start token
        tgt = self.query_start.expand(B, 1, -1)
        loss = 0.0
        mask_visited = torch.zeros(B, N, dtype=torch.bool, device=device)
        valid_steps = 0
        last_node_idx = None  # No previous node for first step

        # Debug prints
        if torch.rand(1).item() < 0.01:  # Print 1% of the time
            print(f"DEBUG: B={B}, N={N}")
            print(f"DEBUG: target_idx shape: {target_idx.shape}, range: [{target_idx.min()}, {target_idx.max()}]")

        for t in range(N):
            logits = self.decode_step(H, tgt, mask_visited, edge_feats=edge_feats, last_node_idx=last_node_idx)
            
            # Verify shapes
            assert logits.shape == (B, N), f"Logits shape: {logits.shape}, expected ({B}, {N})"
            
            y = target_idx[:, t]  # (B,)
            assert y.shape == (B,), f"Target shape: {y.shape}, expected ({B},)"
            
            # Simple cross-entropy loss
            step_loss = F.cross_entropy(logits, y, reduction='mean')
            
            # Skip if loss is invalid
            if torch.isfinite(step_loss):
                loss += step_loss
                valid_steps += 1
            else:
                print(f"Warning: Invalid loss at step {t}: {step_loss.item()}")

            chosen = y
            mask_visited = mask_visited.scatter(1, chosen.unsqueeze(1), True)
            next_embed = H[torch.arange(B, device=device), chosen]
            tgt = torch.cat([tgt, next_embed.unsqueeze(1)], dim=1)
            
            # Update last_node_idx for next iteration
            last_node_idx = chosen

        if valid_steps == 0:
            return torch.tensor(1e6, device=device, requires_grad=True)
        
        avg_loss = loss / valid_steps
        
        # Debug: print loss occasionally
        if torch.rand(1).item() < 0.01:
            print(f"DEBUG: Avg loss: {avg_loss.item():.4f}, valid_steps: {valid_steps}/{N}")
        
        return avg_loss

    @torch.no_grad()

    @torch.no_grad()
    def greedy_decode(self, x, edge_feats=None):
        B, N, _ = x.shape
        device = x.device

        H = self.encode(x)
        tgt = self.query_start.expand(B, 1, -1)
        mask_visited = torch.zeros(B, N, dtype=torch.bool, device=device)
        seq = []
        last_node_idx = None  # No previous node for first step

        for step in range(N):
            logits = self.decode_step(H, tgt, mask_visited, edge_feats=edge_feats, last_node_idx=last_node_idx)
            
            choice = torch.argmax(logits, dim=1)  # (B,)
            seq.append(choice)
            
            # Update mask
            mask_visited = mask_visited.scatter(1, choice.unsqueeze(1), True)
            
            # Get next embedding
            next_embed = H[torch.arange(B, device=device), choice]  # (B,D)
            tgt = torch.cat([tgt, next_embed.unsqueeze(1)], dim=1)
            
            # Update last_node_idx for next iteration
            last_node_idx = choice
            
            # Safety check
            if mask_visited.all():
                break

        # Stack sequence
        seq_tensor = torch.stack(seq, dim=1)  # (B, N)
        
        return seq_tensor