from scipy.stats import wasserstein_distance
from math import radians, cos, sin, asin, sqrt
import math
import torch
import numpy as np
import torch.nn.parallel
from torch_geometric.utils import to_dense_adj

def deg_to_rad(x):
  return x * math.pi / 180

def latlon_to_cart(lat,lon):
  x = np.cos(lat) * np.cos(lon)
  y = np.cos(lat) * np.sin(lon)
  z = np.sin(lat)
  cart_coord = np.column_stack((x, y, z))
  return cart_coord

def haversine(lon1, lat1, lon2, lat2): 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 
    return c * r * 1000

# Helper function for 2+d distance
def newDistance(a, b, nd_dist="great_circle"):  
    # Distance options are ["great_circle" (2D only), "euclidean", "wasserstein" (for higher-dimensional coordinate embeddings)]
    if a.shape[0]==2:
      x1, y1 = a[0], a[1]
      x2, y2 = b[0], b[1]
      if nd_dist=="euclidean":
        d = math.sqrt( ((x1-x2)**2)+((y1-y2)**2))
      else:
        d = haversine(x1,y1,x2,y2)
    if a.shape[0]==3:
      x1, y1, z1 = a[0], a[1], a[2]
      x2, y2, z2 = b[0], b[1], b[2]
      d = math.sqrt(math.pow(x2 - x1, 2) +
                  math.pow(y2 - y1, 2) +
                  math.pow(z2 - z1, 2)* 1.0) 
    if a.shape[0]>3:
      if nd_dist=="wasserstein":
        d = wasserstein_distance(a.reshape(-1).detach(),b.reshape(-1).detach())
        #d = sgw_cpu(a.reshape(1,-1).detach(),b.reshape(1,-1).detach())
      else:
        d = torch.pow(a.reshape(1,1,-1) - b.reshape(1,1,-1), 2).sum(2) 
    return d 

# Helper function for edge weights
def makeEdgeWeight(x, edge_index):
  a = x[edge_index]
  dist = haversine(a[0,:,0], a[0,:,1], a[1,:,0], a[1,:,1])
  # dist = np.linalg.norm(a[0] - a[1], axis=1) # Euclidean
  max_val = dist.max()
  rng = max_val - dist.min()
  edge_weight = (max_val - dist) / rng
  return torch.Tensor(edge_weight)

# knn graph to adjacency matrix (probably already built)
def knn_to_adj(knn, n):
  return to_dense_adj(knn.flip(0)).squeeze()

def normal_torch(tensor,min_val=0):
  t_min = torch.min(tensor)
  t_max = torch.max(tensor)
  if t_min == 0 and t_max == 0:
    return torch.tensor(tensor)
  if min_val == -1:
    tensor_norm = 2 * ((tensor - t_min) / (t_max - t_min)) - 1
  if min_val== 0:
    tensor_norm = ((tensor - t_min) / (t_max - t_min))
  return torch.tensor(tensor_norm)

def lw_tensor_local_moran(y,w_sparse,na_to_zero=True,norm=True,norm_min_val=0):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  y = y.reshape(-1)
  n = len(y)
  n_1 = n - 1
  z = y - y.mean()
  sy = y.std()
  z /= sy
  den = (z * z).sum()
  zl = torch.tensor(w_sparse * z).to(device)
  mi = n_1 * z * zl / den
  if na_to_zero==True:
    mi[torch.isnan(mi)] = 0
  if norm==True:
    mi = normal_torch(mi,min_val=norm_min_val)
  return torch.tensor(mi)
