# src/utils/features.py



import math, torch

def normalize_xy(coords):  # coords: (B,N,2) lat,lon in degrees

    # center by centroid and scale by std for stability

    mean = coords.mean(dim=1, keepdim=True)

    std = coords.std(dim=1, keepdim=True) + 1e-6

    return (coords - mean) / std

def polar_feats(coords):  # (B,N,2) deg

    # convert to rough polar around centroid

    centered = coords - coords.mean(dim=1, keepdim=True)

    x = centered[...,1] * math.pi/180.0  # lon diff (approx)

    y = centered[...,0] * math.pi/180.0  # lat diff (approx)

    r = torch.sqrt(x*x + y*y)

    theta = torch.atan2(y, x)

    return torch.stack([r, theta], dim=-1)

def node_features(coords, aux=None):

    """

    coords: (B,N,2) deg

    aux: optional (B,N,Faux)

    Returns (B,N,F) with [norm_latlon(2), polar(2), aux?]

    """

    norm = normalize_xy(coords)

    pol = polar_feats(coords)

    if aux is None:

        return torch.cat([norm, pol], dim=-1)

    return torch.cat([norm, pol, aux], dim=-1)

def edge_bias_features(coords):

    """

    coords: (B,N,2) deg

    Returns (B,N,N,3) with [distance_km, delta_theta, same_street(=0 placeholder)]

    """

    B,N,_ = coords.shape

    # rough planar distance in km (ok for bias)

    # use radians for angular difference

    rad = coords * (math.pi/180.0)

    lat = rad[...,0].unsqueeze(2)

    lon = rad[...,1].unsqueeze(2)

    dlat = lat - lat.transpose(1,2)

    dlon = lon - lon.transpose(1,2)

    # Haversine small-angle approx for bias; exact not necessary

    dist = torch.sqrt((111.0*dlat)**2 + (111.0*torch.cos(lat)*dlon)**2)  # km approx
    dist = torch.clamp(dist, min=1e-6)  # Prevent division by zero when stops have identical coordinates

    theta = torch.atan2(dlat, dlon+1e-9)

    dtheta = theta - theta.transpose(1,2)

    same = torch.zeros((B,N,N), device=coords.device)  # placeholder; can be filled if street data exists

    return torch.stack([dist, dtheta, same], dim=-1)

