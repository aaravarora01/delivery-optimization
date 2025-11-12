# src/utils/beam_search.py



import math, torch

@torch.no_grad()

def beam_search_pointer(model, x, edge_feats=None, beam_size=4):

    """

    x: (B,N,F)

    Returns best sequences (B,N) and logprob.

    """

    device = x.device

    B,N,_ = x.shape

    H = model.encode(x)

    start = model.query_start.expand(B,1,-1)

    # beams: list of (tgt, mask_visited, seq, logp)

    beams = [(start, torch.zeros(B,N, dtype=torch.bool, device=device), [], torch.zeros(B, device=device))]

    for t in range(N):

        new_beams = []

        for (tgt, mv, seq, logp) in beams:

            logits = model.decode_step(H, tgt, mv, edge_feats=edge_feats)

            logprobs = torch.log_softmax(logits, dim=1)  # (B,N)

            topk = torch.topk(logprobs, k=min(beam_size, N - t), dim=1)

            for k in range(topk.indices.size(1)):

                idx = topk.indices[:,k]  # (B,)

                lp = logp + topk.values[:,k]

                mv_new = mv.clone()

                mv_new.scatter_(1, idx.unsqueeze(1), True)

                next_embed = H[torch.arange(B, device=device), idx]

                tgt_new = torch.cat([tgt, next_embed.unsqueeze(1)], dim=1)

                seq_new = seq + [idx]

                new_beams.append((tgt_new, mv_new, seq_new, lp))

        # prune beams by total logprob

        new_beams.sort(key=lambda b: b[3].sum().item(), reverse=True)

        beams = new_beams[:beam_size]

    # pick best

    best = max(beams, key=lambda b: b[3].sum().item())

    seq_idx = torch.stack(best[2], dim=1)  # (B,N)

    return seq_idx, best[3]

