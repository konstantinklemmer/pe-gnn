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


import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import torch.utils.data
import math

def train(args):
  # Get args
  dset = args.dset
  sphere_transform = args.sphere_trans
  model_name = args.model_name
  random_state = args.random_state
  path = args.path
  train_size = args.train_size
  batched_training = args.batched_training
  batch_size = args.batch_size
  n_epochs = args.n_epochs
  train_crit = args.train_crit
  lr = args.lr
  emb_dim = args.emb_dim
  MAT = args.mat
  uw = args.uw
  lamb = args.lamb
  k = args.k
  save_freq = args.save_freq
  print_progress = args.print_progress

  # Set random seed
  np.random.seed(random_state)

  # Set device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Access and process data
  if dset=="cali_housing":
    c,x,y = get_cali_housing_data()
  if dset=="election":
    c,x,y = get_election_data()
  if dset=="air_temp":
    c,x,y = get_air_temp_data()
  if dset=="3d_road":
    c,x,y = get_3d_road_data()
    x = torch.ones(y.shape[0],1)

  if sphere_transform==True:
    c = norm_sphere(c)

  n = x.shape[0]
  n_train = np.round(n * train_size).astype(int)
  n_test = (n - n_train).astype(int)
  indices = np.arange(n)
  _, _, _, _, idx_train, idx_test = train_test_split(x, y, indices, test_size=(1-train_size), random_state=random_state)

  train_x, test_x = x[idx_train], x[idx_test]
  train_y, test_y = y[idx_train], y[idx_test]
  train_c, test_c = c[idx_train], c[idx_test]
  train_dataset, test_dataset = MyDataset(train_x, train_y, train_c), MyDataset(test_x, test_y, test_c)
  
  if batched_training==False:
    batch_size = len(idx_train)
    train_edge_index = knn_graph(train_c, k=k).to(device)
    train_edge_weight = makeEdgeWeight(train_c, train_edge_index).to(device)
    test_edge_index = knn_graph(test_c, k=k).to(device)
    test_edge_weight = makeEdgeWeight(test_c, test_edge_index).to(device)
    train_moran_weight_matrix = knn_to_adj(train_edge_index, batch_size) #libpysal.weights.KNN(batch_y.cpu(), k=20).to(device)
    with torch.enable_grad():
      train_y_moran = lw_tensor_local_moran(train_y, sparse.csr_matrix(train_moran_weight_matrix)).to(device)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False, drop_last=False)
  else:
    train_edge_index = False
    train_edge_weight = False
    test_edge_index = False
    test_edge_weight = False
    train_y_moran = False
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, drop_last=True)

  # Make model
  if model_name=="gcn":
    model = GCN(num_features_in=train_x.shape[1],k=k,MAT=MAT).to(device)
  if model_name=="pegcn":
    model = PEGCN(num_features_in=train_x.shape[1],k=k,MAT=MAT,emb_dim=emb_dim).to(device)
  model = model.float()

  if MAT:
    task_num = 2
  else:
    task_num = 1

  loss_wrapper = LossWrapper(model, task_num=task_num, loss=train_crit, uw=uw, lamb=lamb, k=k, batch_size=batch_size).to(device)
  optimizer = torch.optim.Adam(loss_wrapper.parameters(), lr=lr)
  score1 = nn.MSELoss()
  score2 = nn.L1Loss()
  
  # Tensorboard and logging
  test_ = dset + '-' + model_name + '-k' + str(k)
  if model_name=='pegcn':
    test_ = test_ + '-emb' + str(emb_dim)
  if MAT:
    if uw:
      test_ = test_ + "_mat-uw"
    else:
      test_ = test_ + "_mat-lam" + str(lamb)
  if batched_training==True:
    test_ = test_ + "_bs" + str(batch_size) + "_ep" + str(n_epochs)
  else:
    test_ = test_ + "_bsn_ep" + str(n_epochs)

  saved_file = "{}_{}{}-{}:{}:{}.{}".format(test_,
                                            datetime.now().strftime("%h"),
                                            datetime.now().strftime("%d"),
                                            datetime.now().strftime("%H"),
                                            datetime.now().strftime("%M"),
                                            datetime.now().strftime("%S"),
                                            datetime.now().strftime("%f"))

  log_dir = path + "/trained/{}/log".format(saved_file)

  if not os.path.exists(path + "trained/{}/data".format(saved_file)):
      os.makedirs(path + "/trained/{}/data".format(saved_file))
  if not os.path.exists(path + "/trained/{}/images".format(saved_file)):
      os.makedirs(path + "/trained/{}/images".format(saved_file))
  with open(path + "/trained/{}/train_notes.txt".format(saved_file), 'w') as f:
      # Include any experiment notes here:
      f.write("Experiment notes: .... \n\n")
      f.write("MODEL_DATA: {}\n".format(
          test_))
      f.write("BATCH_SIZE: {}\nLEARNING_RATE: {}\n".format(
          batch_size,
          lr))
  
  writer = SummaryWriter(log_dir)
  
  # Training loop
  it_counts = 0
  for epoch in range(n_epochs):
    for batch in train_loader:
      model.train()
      it_counts += 1
      x = batch[0].to(device).float()
      y = batch[1].to(device).float()
      c = batch[2].to(device).float()

      optimizer.zero_grad()

      if MAT==True & uw==True:
        loss, log_vars = loss_wrapper(x, y, c, train_edge_index, train_edge_weight, train_y_moran)
      else:
        loss = loss_wrapper(x, y, c, train_edge_index, train_edge_weight, train_y_moran)
      loss.backward()
      optimizer.step()
      # Eval 
      if it_counts % save_freq == 0:
        model.eval()
        with torch.no_grad():
          if MAT:
            pred,_ = model(torch.tensor(test_dataset.features).to(device),torch.tensor(test_dataset.coords).to(device),test_edge_index,test_edge_weight)
          else:
            pred = model(torch.tensor(test_dataset.features).to(device),torch.tensor(test_dataset.coords).to(device),test_edge_index,test_edge_weight)
        test_score1 = score1(torch.tensor(test_dataset.target).reshape(-1).to(device), pred.reshape(-1))
        test_score2 = score2(torch.tensor(test_dataset.target).reshape(-1).to(device), pred.reshape(-1))

        if print_progress:
          print("Epoch [%d/%d] - Loss: %f - Test score (MSE): %f - Test score (MAE): %f" % (epoch, n_epochs, loss.item(), test_score1.item(), test_score2.item()))
        save_path = path + "/trained/{}/ckpts".format(saved_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model, save_path + '/' + 'model.pt')
        writer.add_scalar('Test score (MSE)', test_score1.item(), it_counts)
        writer.add_scalar('Test score (MAE)', test_score2.item(), it_counts)
      writer.add_scalar('Training loss', loss.item(), it_counts)
      if MAT==True & uw==True:
        writer.add_scalar('Uncertainty weight: main task', log_vars[0], it_counts)
        writer.add_scalar('Uncertainty weight: Morans aux task', log_vars[1], it_counts)  
      writer.flush()
  print("Saved all models to {}".format(save_path))
  print("Epoch [%d/%d] - Loss: %f - Test score (MSE): %f - Test score (MAE): %f" % (epoch, n_epochs, loss.item(), test_score1.item(), test_score2.item()))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GCN Kriging')
  # Data & model selection
  parser.add_argument('-d', '--dset', type=str, default='election',
                      choices=['cali_housing', 'election', 'air_temp', '3d_road'])
  parser.add_argument('-m', '--model_name',  type=str, default='gcn', choices=['gcn','pegcn'])
  parser.add_argument('-st', '--sphere_trans', type=bool, default=False)
  # Utilities
  parser.add_argument('-s', '--random_state', type=int, default=1)
  parser.add_argument('-p', '--path', type=str, default='./')
  # Training setting 
  parser.add_argument('-ts', '--train_size', type=float, default=0.8)
  parser.add_argument('-bt', '--batched_training', type=bool, default=True)
  parser.add_argument('-bs', '--batch_size', type=int, default=1024)
  parser.add_argument('-ne', '--n_epochs', type=int, default=50)
  parser.add_argument('-loss', '--train_crit', type=str, default='mse', choices=['mse','l1'])
  parser.add_argument('-lr', '--lr', type=float, default=1e-3)
  # Model config
  parser.add_argument('-embd', '--emb_dim', type=float, default=64)
  parser.add_argument('-mat', '--mat', type=bool, default=False)
  parser.add_argument('-u', '--uw', type=bool, default=False)
  parser.add_argument('-l', '--lamb', type=float, default=0.5)
  parser.add_argument('-k', '--k', type=int, default=5)
  # Logging & evaluation
  parser.add_argument('-save', '--save_freq', type=int, default=5)
  parser.add_argument('-print', '--print_progress', type=bool, default=True)
  parser.add_argument('-f') #Dummy to get parser to run in Colab

  args = parser.parse_args()

  out = train(args)