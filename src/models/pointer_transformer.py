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

        # DEBUG: Check inputs
        print(f"\n=== DEBUG decode_step ===")
        print(f"Input H shape: {H.shape}, finite: {torch.isfinite(H).all().item()}, "
              f"min: {H.min().item():.6f}, max: {H.max().item():.6f}, "
              f"mean: {H.mean().item():.6f}, std: {H.std().item():.6f}")
        assert torch.isfinite(H).all(), f"NaN/inf in H (encoder output)! Count: {(~torch.isfinite(H)).sum().item()}"
        
        print(f"Input tgt shape: {tgt.shape}, finite: {torch.isfinite(tgt).all().item()}, "
              f"min: {tgt.min().item():.6f}, max: {tgt.max().item():.6f}, "
              f"mean: {tgt.mean().item():.6f}, std: {tgt.std().item():.6f}")
        assert torch.isfinite(tgt).all(), f"NaN/inf in tgt (decoder input)! Count: {(~torch.isfinite(tgt)).sum().item()}"
        
        if edge_feats is not None:
            print(f"Input edge_feats shape: {edge_feats.shape}, finite: {torch.isfinite(edge_feats).all().item()}, "
                  f"min: {edge_feats.min().item():.6f}, max: {edge_feats.max().item():.6f}")
            assert torch.isfinite(edge_feats).all(), f"NaN/inf in edge_feats! Count: {(~torch.isfinite(edge_feats)).sum().item()}"
        
        print(f"mask_visited shape: {mask_visited.shape}, sum: {mask_visited.sum().item()}/{mask_visited.numel()}")

        # standard transformer decoding
        T = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()

        D = self.decoder(tgt, H, tgt_mask=tgt_mask)  # (B,T,D)
        
        # DEBUG: Check decoder output
        print(f"Decoder output D shape: {D.shape}, finite: {torch.isfinite(D).all().item()}, "
              f"min: {D.min().item():.6f}, max: {D.max().item():.6f}, "
              f"mean: {D.mean().item():.6f}, std: {D.std().item():.6f}")
        assert torch.isfinite(D).all(), f"NaN/inf in decoder output D! Count: {(~torch.isfinite(D)).sum().item()}"
        
        # Clamp for numerical stability
        D = torch.clamp(D, min=-100.0, max=100.0)

        q = self.out_proj(D[:, -1])  # (B,D) last step
        
        # DEBUG: Check query after projection
        print(f"Query q (after out_proj) shape: {q.shape}, finite: {torch.isfinite(q).all().item()}, "
              f"min: {q.min().item():.6f}, max: {q.max().item():.6f}, "
              f"mean: {q.mean().item():.6f}, std: {q.std().item():.6f}")
        assert torch.isfinite(q).all(), f"NaN/inf in query q after out_proj! Count: {(~torch.isfinite(q)).sum().item()}"
        
        q = torch.clamp(q, min=-10.0, max=10.0)
        
        # Normalize query to prevent extreme values
        q_norm = torch.norm(q, dim=-1, keepdim=True) + 1e-8
        print(f"q_norm: min={q_norm.min().item():.6f}, max={q_norm.max().item():.6f}, mean={q_norm.mean().item():.6f}")
        if (q_norm < 1e-7).any():
            print(f"  ⚠️  q_norm is very small! min={q_norm.min().item():.10f}")
        
        q = q / q_norm
        
        # DEBUG: Check after normalization
        print(f"Query q (after norm) shape: {q.shape}, finite: {torch.isfinite(q).all().item()}, "
              f"min: {q.min().item():.6f}, max: {q.max().item():.6f}, "
              f"mean: {q.mean().item():.6f}, std: {q.std().item():.6f}")
        assert torch.isfinite(q).all(), f"NaN/inf in query q after normalization! Count: {(~torch.isfinite(q)).sum().item()}"

        keys = self.ptr_proj(H)      # (B,N,D)
        
        # DEBUG: Check keys after projection
        print(f"Keys (after ptr_proj) shape: {keys.shape}, finite: {torch.isfinite(keys).all().item()}, "
              f"min: {keys.min().item():.6f}, max: {keys.max().item():.6f}, "
              f"mean: {keys.mean().item():.6f}, std: {keys.std().item():.6f}")
        assert torch.isfinite(keys).all(), f"NaN/inf in keys after ptr_proj! Count: {(~torch.isfinite(keys)).sum().item()}"
        
        keys = torch.clamp(keys, min=-10.0, max=10.0)
        
        # Normalize keys to prevent extreme values
        keys_norm = torch.norm(keys, dim=-1, keepdim=True) + 1e-8
        print(f"keys_norm: min={keys_norm.min().item():.6f}, max={keys_norm.max().item():.6f}, mean={keys_norm.mean().item():.6f}")
        if (keys_norm < 1e-7).any():
            print(f"  ⚠️  keys_norm is very small! min={keys_norm.min().item():.10f}")
        
        keys = keys / keys_norm
        
        # DEBUG: Check keys after normalization
        print(f"Keys (after norm) shape: {keys.shape}, finite: {torch.isfinite(keys).all().item()}, "
              f"min: {keys.min().item():.6f}, max: {keys.max().item():.6f}")
        assert torch.isfinite(keys).all(), f"NaN/inf in keys after normalization! Count: {(~torch.isfinite(keys)).sum().item()}"

        # Einsum with epsilon in sqrt to prevent division issues
        scale = math.sqrt(keys.size(-1)) + 1e-8
        logits = torch.einsum("bd,bnd->bn", q, keys) / scale  # pointer scores
        
        # DEBUG: Check logits after einsum
        print(f"Logits (after einsum) shape: {logits.shape}, finite: {torch.isfinite(logits).all().item()}, "
              f"min: {logits.min().item():.6f}, max: {logits.max().item():.6f}, "
              f"mean: {logits.mean().item():.6f}, std: {logits.std().item():.6f}")
        assert torch.isfinite(logits).all(), f"NaN/inf in logits after einsum! Count: {(~torch.isfinite(logits)).sum().item()}"
        
        # Clamp logits for numerical stability
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        if self.edge_bias is not None and edge_feats is not None:
            # Use last chosen index to index edge_feats for bias against candidates
            # We approximate with a pooled bias: bias_i = mean_j(bias_ij) to stay cheap
            bias = self.edge_bias(edge_feats)  # (B,N,N)
            
            # DEBUG: Check bias after edge_bias network
            print(f"Bias (after edge_bias) shape: {bias.shape}, finite: {torch.isfinite(bias).all().item()}, "
                  f"min: {bias.min().item():.6f}, max: {bias.max().item():.6f}, "
                  f"mean: {bias.mean().item():.6f}, std: {bias.std().item():.6f}")
            assert torch.isfinite(bias).all(), f"NaN/inf in bias after edge_bias network! Count: {(~torch.isfinite(bias)).sum().item()}"
            
            bias = bias.mean(dim=1)            # (B,N)
            
            # DEBUG: Check bias after mean
            print(f"Bias (after mean) shape: {bias.shape}, finite: {torch.isfinite(bias).all().item()}, "
                  f"min: {bias.min().item():.6f}, max: {bias.max().item():.6f}, "
                  f"mean: {bias.mean().item():.6f}, std: {bias.std().item():.6f}")
            assert torch.isfinite(bias).all(), f"NaN/inf in bias after mean! Count: {(~torch.isfinite(bias)).sum().item()}"
            
            # Clamp bias after mean to prevent extreme values
            bias = torch.clamp(bias, min=-10.0, max=10.0)

            logits = logits + bias
            
            # DEBUG: Check logits after adding bias
            print(f"Logits (after adding bias) shape: {logits.shape}, finite: {torch.isfinite(logits).all().item()}, "
                  f"min: {logits.min().item():.6f}, max: {logits.max().item():.6f}, "
                  f"mean: {logits.mean().item():.6f}, std: {logits.std().item():.6f}")
            assert torch.isfinite(logits).all(), f"NaN/inf in logits after adding bias! Count: {(~torch.isfinite(logits)).sum().item()}"

        logits = logits.masked_fill(mask_visited, float(-1e9))
        
        # DEBUG: Final logits check (note: masked values will be -1e9, not finite check)
        print(f"Final logits shape: {logits.shape}")
        print(f"  Non-masked logits: min={logits[~mask_visited].min().item():.6f}, max={logits[~mask_visited].max().item():.6f}, "
              f"mean={logits[~mask_visited].mean().item():.6f}, std={logits[~mask_visited].std().item():.6f}")
        print(f"  Masked count: {mask_visited.sum().item()}/{mask_visited.numel()}")
        # Check non-masked values are finite
        if (~mask_visited).any():
            assert torch.isfinite(logits[~mask_visited]).all(), f"NaN/inf in non-masked final logits! Count: {(~torch.isfinite(logits[~mask_visited])).sum().item()}"
        print(f"=== END DEBUG decode_step ===\n")

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

        H = self.encode(x)
        tgt = self.query_start.expand(B,1,-1)
        loss = 0.0
        mask_visited = torch.zeros(B, N, dtype=torch.bool, device=device)

        for t in range(N):
            # Get pointer distribution over N input nodes
            logits = self.decode_step(H, tgt, mask_visited, edge_feats=edge_feats)
            
            y = target_idx[:, t]  # (B,) - target node index at step t
            
            # Standard cross-entropy over pointer distribution (no label smoothing)
            # This treats it as a classification problem: which of the N nodes to select?
            step_loss = F.cross_entropy(logits, y, reduction='mean')
            
            loss += step_loss

            # Teacher forcing: use target node for next step
            chosen = y
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

