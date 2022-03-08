from scipy.stats import wasserstein_distance
from math import radians, cos, sin, asin, sqrt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import Sampler
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, knn_graph
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from geopy.distance import distance
import math
import matplotlib.pyplot as plt

import os
import datetime
import sys
import requests
from urllib.request import urlretrieve
import urllib.request, json 
import zipfile
import subprocess

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from decimal import Decimal, getcontext
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, IterableDataset, DataLoader

import numpy as np
import pandas as pd
import scipy
from scipy import sparse

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize
from sklearn import metrics

import torch
import argparse
import glob
import os
import time
import tqdm

from datetime import datetime
import numpy as np
from urllib.request import urlretrieve
import urllib.request, json
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
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
  to = edge_index[0]
  fro = edge_index[1]
  edge_weight = []
  for i in range(len(to)):
    edge_weight.append(newDistance(x[to[i]],x[fro[i]])) # probably want to do inverse distance eventually
  max_val = max(edge_weight)
  rng = max_val - min(edge_weight)
  edge_weight = [(max_val - elem) / rng for elem in edge_weight]
  return torch.Tensor(edge_weight)

# knn graph to adjacency matrix (probably already built)
def knn_to_adj(knn, n):
  adj_matrix = torch.zeros(n, n, dtype=float) #lil_matrix((n, n), dtype=float) 
  for i in range(len(knn[0])):
    tow = knn[0][i]
    fro = knn[1][i]
    adj_matrix[tow,fro] = 1 # should be bidectional?
  return adj_matrix.T

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