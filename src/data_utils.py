import io
import requests
from urllib import request 
from zipfile import ZipFile
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import sklearn.datasets
from functools import reduce

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

def normal(x,min_val=0):
  '''
  Normalize a vector

  Parameters:
  x = numerical vector
  min_val = integer; choice of [0,1], setting whether normalization in range[0,1] or [-1,1]

  Return:
  x_norm = normalized vector
  '''
  x_min = np.min(x)
  x_max = np.max(x)
  if x_min == 0 and x_max == 0:
    return x
  if min_val == -1:
    x_norm = 2 * ((x - x_min) / (x_max - x_min)) - 1
  if min_val== 0:
    x_norm = ((x - x_min) / (x_max - x_min))
  return x_norm

def get_election_data(pred="gop_2016",norm_x=True,norm_y=True,norm_min_val=0,spat_int=True):
  '''
  Download and process the Election dataset used in CorrelationGNN (https://arxiv.org/abs/2002.08274)

  Parameters:
  pred = numeric; outcome variable to be returned; choose from ["dem_2016",
                                                       "gop_2016",
                                                       "MedianIncome2016",
                                                       "R_NET_MIG_2016",
                                                       "R_birth_2016",
                                                       "R_death_2016",
                                                       "BachelorRate2016",
                                                       "Unemployment_rate_2016"]
  norm_x = logical; should features be normalized
  norm_y = logical; should outcome be normalized
  norm_min_val = integer; choice of [0,1], setting whether normalization in range[0,1] or [-1,1]

  Return:
  coords = spatial coordinates (lon/lat)
  x = features at location (excluding outcome variable)
  y = outcome variable
  '''

  Path("./election_data").mkdir(parents=True, exist_ok=True)
  zipurl = 'https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip'
  with request.urlopen(zipurl) as zipresp:
      with ZipFile(io.BytesIO(zipresp.read())) as zfile:
          zfile.extractall('./election_data')

  geo = pd.read_csv("./election_data/2020_Gaz_counties_national.txt",sep='\t')
  geo = geo.rename(columns={"GEOID":"FIPS",'INTPTLONG                                                                                                               ':'INTPTLONG'})

  url = 'https://raw.githubusercontent.com/000Justin000/gnn-residual-correlation/master/datasets/election/education.csv'
  url_open = request.urlopen(url)
  edu = pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))) 

  url = 'https://raw.githubusercontent.com/000Justin000/gnn-residual-correlation/master/datasets/election/election.csv'
  url_open = request.urlopen(url)
  ele = pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))) 
  ele = ele.rename(columns={"fips_code":"FIPS"})

  url = 'https://raw.githubusercontent.com/000Justin000/gnn-residual-correlation/master/datasets/election/income.csv'
  url_open = request.urlopen(url)
  inc = pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))) 

  url = 'https://raw.githubusercontent.com/000Justin000/gnn-residual-correlation/master/datasets/election/unemployment.csv'
  url_open = request.urlopen(url)
  une = pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))) 

  url = 'https://raw.githubusercontent.com/000Justin000/gnn-residual-correlation/master/datasets/election/population.csv'
  url_open = request.urlopen(url)
  pop = pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))) 

  dfs = [geo,edu,ele,inc,une,pop]
  data = reduce(lambda  left,right: pd.merge(left,right,on=['FIPS'],how='outer'), dfs)
  data = data.replace({',':''}, regex=True)

  out_data = np.array([data.INTPTLONG,data.INTPTLAT,data.dem_2016,data.gop_2016,data.MedianIncome2016,data.R_NET_MIG_2016,data.R_birth_2016,data.R_death_2016,data.BachelorRate2016,data.Unemployment_rate_2016]).T.astype(float)
  out_data = out_data[~np.isnan(out_data).any(axis=1)]
  out_data = out_data[(out_data[:,0] > -130) & (out_data[:,0] < -50) & (out_data[:,1] > 22) & (out_data[:,1] < 50)]

  coords = out_data[:,:2]
  if pred == "dem_2016":
    y = out_data[:,2]
    x = out_data[:,3:]
  if pred == "gop_2016":
    y = out_data[:,3]
    x = out_data[:,[2,4,5,6,7,8,9]]
  if pred == "MedianIncome2016":
    y = out_data[:,4]
    x = out_data[:,[2,3,5,6,7,8,9]]
  if pred == "R_NET_MIG_2016":
    y = out_data[:,5]
    x = out_data[:,[2,3,4,6,7,8,9]]
  if pred == "R_birth_2016":
    y = out_data[:,6]
    x = out_data[:,[2,3,4,5,7,8,9]]
  if pred == "R_death_2016":
    y = out_data[:,7]
    x = out_data[:,[2,3,4,5,6,8,9]]
  if pred == "BachelorRate2016":
    y = out_data[:,8]
    x = out_data[:,[2,3,4,5,6,7,9]]
  if pred == "Unemployment_rate_2016":
    y = out_data[:,9]
    x = out_data[:,[2,3,4,5,6,7,8]]

  if norm_y==True:
    y = normal(y,norm_min_val)
  if norm_x==True:
    for i in range(x.shape[1]):
      x[:,i] = normal(x[:,i],norm_min_val)
  if spat_int==True:
      x = torch.ones(x.shape[0],1)
  return torch.tensor(coords), torch.tensor(x), torch.tensor(y)

def get_cali_housing_data(norm_x=True,norm_y=True,norm_min_val=0,spat_int=False):
  '''
  Download and process the California Housing Dataset

  Parameters:
  norm_x = logical; should features be normalized
  norm_y = logical; should outcome be normalized
  norm_min_val = integer; choice of [0,1], setting whether normalization in range[0,1] or [-1,1]

  Return:
  coords = spatial coordinates (lon/lat)
  x = features at location
  y = outcome variable
  '''
  cali_housing_ds = sklearn.datasets.fetch_california_housing()
  coords = np.array(cali_housing_ds.data[:,6:])
  y = np.array(cali_housing_ds.target)
  x = np.array(cali_housing_ds.data[:,:6])
  if norm_y==True:
    y = normal(y,norm_min_val)
  if norm_x==True:
    for i in range(x.shape[1]):
      x[:,i] = normal(x[:,i],norm_min_val)
  if spat_int==True:
    x = torch.ones(x.shape[0],1)
  return torch.tensor(coords), torch.tensor(x), torch.tensor(y)

def get_3d_road_data(norm_y=True,norm_min_val=0):
  '''
  Download and process the 3d road dataset

  Parameters:
  norm_x = logical; should features be normalized
  norm_y = logical; should outcome be normalized
  norm_min_val = integer; choice of [0,1], setting whether normalization in range[0,1] or [-1,1]

  Return:
  coords = spatial coordinates (lon/lat)
  x = features at location; empty for this dataset
  y = outcome variable
  '''
  # Both of the above sources contain the 3d road dataset
  #url="https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt"
  url="http://nrvis.com/data/mldata/3D_spatial_network.csv"
  s=requests.get(url).content
  c=pd.read_csv(io.StringIO(s.decode('utf-8')))
  c.columns = ["id","x","y","z"]
  coords = np.array(c[["x","y"]])
  y = np.array(c[["z"]])
  if norm_y==True:
    y = normal(y,norm_min_val)

  return torch.tensor(coords), None, torch.tensor(y)

def get_air_temp_data(pred="temp",norm_y=True,norm_x=True,norm_min_val=0):
  '''
  Download and process the Global Air Temperature dataset

  Parameters:
  pred = numeric; outcome variable to be returned; choose from ["temp", "prec"]
  norm_y = logical; should outcome be normalized
  norm_min_val = integer; choice of [0,1], setting whether normalization in range[0,1] or [-1,1]

  Return:
  coords = spatial coordinates (lon/lat)
  x = features at location
  y = outcome variable
  '''
  url = 'https://springernature.figshare.com/ndownloader/files/12609182'
  url_open = request.urlopen(url)
  inc = np.array(pd.read_csv(io.StringIO(url_open.read().decode('utf-8'))))
  coords = inc[:,:2]
  if pred=="temp":
    y = inc[:,4].reshape(-1)
    x = inc[:,5]
  else:
    y = inc[:,5].reshape(-1)
    x = inc[:,4]
  if norm_y==True:
    y = normal(y,norm_min_val)
  if norm_x==True:
    x = normal(x,norm_min_val).reshape(-1,1)

  return torch.tensor(coords), torch.tensor(x), torch.tensor(y)

class MyDataset(Dataset):
    def __init__(self, x, y, coords):
      self.features = x
      self.target = y
      self.coords = coords
    def __len__(self):
      return len(self.features)
    def __getitem__(self, idx):
      return torch.tensor(self.features[idx]), torch.tensor(self.target[idx]), torch.tensor(self.coords[idx])