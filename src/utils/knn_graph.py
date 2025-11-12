# src/utils/knn_graph.py



import math, torch

def haversine_torch(latlon):  # (B,N,2) in radians

    # pairwise distances in km

    B,N,_ = latlon.shape

    lat = latlon[...,0].unsqueeze(2)

    lon = latlon[...,1].unsqueeze(2)

    dlat = lat - lat.transpose(1,2)

    dlon = lon - lon.transpose(1,2)

    a = (torch.sin(dlat/2)**2 +

         torch.cos(lat).unsqueeze(2)*torch.cos(lat).unsqueeze(1)*torch.sin(dlon/2)**2)

    return 2*6371.0088*torch.arcsin(torch.sqrt(a)+1e-12)

def knn_adj(latlon_deg, k=8):

    """

    latlon_deg: (B,N,2) degrees; returns adj (B,N,N) row-normalized.

    """

    latlon = latlon_deg * (math.pi/180.0)

    D = haversine_torch(latlon)  # (B,N,N)

    # mask self

    B,N,_ = D.shape

    eye = torch.eye(N, device=D.device).bool().unsqueeze(0).expand(B,-1,-1)

    D = D.masked_fill(eye, float('inf'))

    idx = torch.topk(-D, k=k, dim=2).indices  # k nearest (largest -D)

    adj = torch.zeros_like(D)

    rows = torch.arange(N, device=D.device).view(1,N,1).expand(B,N,k)

    adj.scatter_(2, idx, 1.0)

    # include self loops

    adj = adj + eye.float()

    # row normalize

    adj = adj / (adj.sum(dim=2, keepdim=True) + 1e-9)

    return adj

