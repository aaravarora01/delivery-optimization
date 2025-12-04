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

    def decode_step(self, H, tgt, mask_visited, edge_feats=None):

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

        if self.training:
            D = checkpoint.checkpoint(self.decoder, tgt, H, tgt_mask)
        else:
            D = self.decoder(tgt, H, tgt_mask=tgt_mask)
        
        q = self.out_proj(D[:, -1])  # (B,D) last step
        
        keys = self.ptr_proj(H)      # (B,N,D)

        logits = torch.einsum("bd,bnd->bn", q, keys) / math.sqrt(keys.size(-1))  # pointer scores

        if self.edge_bias is not None and edge_feats is not None:
            bias = self.edge_bias(edge_feats)  # (B,N,N)
            bias = bias.mean(dim=1)            # (B,N)
            bias = torch.clamp(bias, min=-10.0, max=10.0)
            logits = logits + bias

        # Clamp logits before masking to prevent overflow
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        logits = logits.masked_fill(mask_visited, -1e4)  # Use -1e4 for FP16 compatibility

        return logits

    def forward_teacher_forced(self, x, target_idx, edge_feats=None):

        """

        x: (B,N,d_in); target_idx: (B,N) indices 0..N-1 of the true sequence

        Returns CE loss (teacher forcing).

        """

        B,N,_ = x.shape

        device = x.device

        # Check input features
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Invalid input x detected")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        H = self.encode(x)
        
        # Start token

        tgt = self.query_start.expand(B,1,-1)

        loss = 0.0

        mask_visited = torch.zeros(B, N, dtype=torch.bool, device=device)

        valid_steps = 0

        for t in range(N):

            logits = self.decode_step(H, tgt, mask_visited, edge_feats=edge_feats)

            y = target_idx[:, t]
            
            # Convert to one-hot for label smoothing
            num_classes = logits.size(-1)
            y_one_hot = torch.zeros(logits.size(0), num_classes, device=logits.device)
            y_one_hot.scatter_(1, y.unsqueeze(1), 1.0)
            
            # Apply label smoothing ONLY to unvisited nodes
            unvisited_mask = ~mask_visited
            smoothing_weight = 0.05
            y_one_hot = y_one_hot * (1 - smoothing_weight) + (smoothing_weight / unvisited_mask.sum(dim=1, keepdim=True).float()) * unvisited_mask.float()
            
            # Compute cross-entropy loss with smoothed one-hot targets
            # No need to clone - we can mask in-place since decode_step already masked
            log_probs = F.log_softmax(logits, dim=-1)
            
            step_loss = -(y_one_hot * log_probs).sum(dim=-1).mean()
            
            # Skip if loss is invalid (should be rare with proper initialization)
            if torch.isfinite(step_loss):
                loss += step_loss
                valid_steps += 1

            chosen = y
            mask_visited = mask_visited.scatter(1, chosen.unsqueeze(1), True)
            next_embed = H[torch.arange(B, device=device), chosen]
            tgt = torch.cat([tgt, next_embed.unsqueeze(1)], dim=1)

        if valid_steps == 0:
            return torch.tensor(1e6, device=device, requires_grad=True)
        
        return loss / valid_steps

    @torch.no_grad()

    def greedy_decode(self, x, edge_feats=None):

        B,N,_ = x.shape

        device = x.device

        H = self.encode(x)

        tgt = self.query_start.expand(B,1,-1)

        mask_visited = torch.zeros(B, N, dtype=torch.bool, device=device)

        seq = []

        for step in range(N):

            logits = self.decode_step(H, tgt, mask_visited, edge_feats=edge_feats)

            choice = torch.argmax(logits, dim=1)  # (B,)
            
            # Convert to scalar for batch size 1, keep tensor for batch > 1
            if B == 1:
                # Explicitly convert to Python int
                if choice.numel() == 1:
                    choice_val = int(choice.item())
                else:
                    choice_val = int(choice[0].item())
                seq.append(choice_val)
                choice_tensor = choice  # Keep tensor for indexing
            else:
                seq.append(choice)
                choice_tensor = choice

            # Update mask - use tensor version for scatter
            mask_visited = mask_visited.scatter(1, choice_tensor.unsqueeze(1), True)

            # Get next embedding - use tensor version for indexing
            next_embed = H[torch.arange(B, device=device), choice_tensor]  # (B,D)

            tgt = torch.cat([tgt, next_embed.unsqueeze(1)], dim=1)
            
            # Safety check: if we've visited all nodes, break early
            if mask_visited.all():
                break

        # Convert to tensor properly - ensure we have exactly N items
        if len(seq) != N:
            print(f"Warning: greedy_decode produced {len(seq)} items but expected {N}")
        
        if B == 1:
            # Ensure seq is a flat list of integers - add debug
            if len(seq) > 0 and not isinstance(seq[0], (int, float)):
                print(f"Error: seq contains non-scalar values! First item type: {type(seq[0])}, value: {seq[0]}")
            
            seq_flat = [int(s) for s in seq]  # Force all to int
            
            # Create tensor explicitly as 1D
            try:
                seq_tensor = torch.tensor(seq_flat, dtype=torch.long, device=device)  # Should be (N,)
            except Exception as e:
                print(f"Error creating tensor from seq_flat: {e}")
                print(f"  seq_flat type: {type(seq_flat)}, length: {len(seq_flat)}")
                print(f"  First few items: {seq_flat[:5]}")
                raise
            
            # Debug: check shape immediately after creation
            if seq_tensor.dim() != 1:
                print(f"Error: seq_tensor has {seq_tensor.dim()} dimensions after creation! Shape: {seq_tensor.shape}")
                print(f"  seq_flat was: {seq_flat[:10]}...")
                # Force flatten
                seq_tensor = seq_tensor.flatten()
            
            if seq_tensor.shape[0] != N:
                print(f"Error: Sequence tensor has shape {seq_tensor.shape}, expected ({N},)")
                print(f"  len(seq)={len(seq)}, N={N}")
                # Truncate or pad to correct size
                if seq_tensor.shape[0] > N:
                    seq_tensor = seq_tensor[:N]
                else:
                    padding = torch.zeros(N - seq_tensor.shape[0], dtype=torch.long, device=device)
                    seq_tensor = torch.cat([seq_tensor, padding])
            
            seq_tensor = seq_tensor.unsqueeze(0)  # (1, N)
            
            # Final check before return
            if seq_tensor.shape != (1, N):
                print(f"Error: Final tensor shape is {seq_tensor.shape}, expected (1, {N})")
                print(f"  Forcing reshape to (1, {N})")
                seq_tensor = seq_tensor.view(1, N)
        else:
            # Handle batch size > 1
            seq_tensor = torch.stack(seq, dim=1)  # (B,N)
        
        print(f"DEBUG greedy_decode returning: shape={seq_tensor.shape}")  # Add this line
        return seq_tensor