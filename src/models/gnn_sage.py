# src/models/gnn_sage.py



from typing import Optional

import torch

import torch.nn as nn

import torch.nn.functional as F

class GraphSAGE(nn.Module):

    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 128, dropout: float = 0.1):

        super().__init__()

        self.lin1 = nn.Linear(in_dim, hidden)

        self.lin2 = nn.Linear(hidden, out_dim)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):

        """

        x: (B,N,d), adj: (B,N,N) binary or weights (row-normalized preferred)

        mean-aggregate neighbors then MLP.

        """

        h1 = self.lin1(x)

        neigh = torch.matmul(adj, h1)  # (B,N,d)

        h = F.relu(h1 + neigh)

        h = self.drop(h)

        h2 = self.lin2(h)

        return h2

