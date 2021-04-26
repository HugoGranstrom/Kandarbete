import csv

import argparse
from concurrent import futures
import os
import re
import sys

import boto3
import botocore
import tqdm

import torch.nn as nn
import torch.nn.functional as F

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import time
from multiprocessing import Process, Queue
import os
import numpy as np

import pandas as pd

import random

from dataset import *
from hqset import *
from net import *
from unet import *
from test import predict

from collections import namedtuple

import torch
from torchvision import models


if __name__ == '__main__':
  torch.multiprocessing.freeze_support()

  csvfile = pd.read_csv("ImageUID.csv", names=["id"])
  ids = csvfile["id"].values
  print(len(ids))

  torch.manual_seed(1337)
    
  print('cuda' if torch.cuda.is_available() else 'cpu')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #device = torch.device('cpu')

  lr_min = 0.0001
  lr_max = 0.0005

  net = UNet(depth=5).to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)
  
  disc = models.vgg16(pretrained=True)
    
  for param in disc.features.parameters():
    param.require_grad = False
      
  num_features = disc.classifier[6].in_features
  disc.classifier[-1] =  nn.Linear(num_features, 1)
  disc.to(device)
  
  optimizer_disc = torch.optim.Adam(disc.parameters(), lr=0.0002)

  criterion = nn.BCELoss()

  real_label = 1.
  fake_label = 0.

  filename = "GAN_UNet_v1.pt"

  iterations, train_losses, val_losses = loadNet(filename, net, optimizer, disc, optimizer_disc, device)
  best_loss = min(val_losses) if len(val_losses) > 0 else 1e6
  print("Best validation loss:", best_loss)
  iteration = iterations[-1] if len(iterations) > 0 else -1
  #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max, step_size_up=2000, last_epoch=iteration, mode="triangular", cycle_momentum=False)

  net.train()
  net.to(device)
  validation_size = 100
  batch_size = 1

  dataset = OpenDataset(ids[:-validation_size], batch_size=batch_size, SUPER_BATCHING=40, high_res_size=(256, 256), low_res_size=(128, 128))
  validation_dataset = OpenDataset(ids[-validation_size:], batch_size=batch_size, SUPER_BATCHING=1, high_res_size=(256, 256), low_res_size=(128, 128))
  validation_data = [i for i in validation_dataset]
  validation_size = len(validation_data)
  """
  traindata = FolderSet("train")
  validdata = FolderSet("valid", length_multiplier=1)

  dataset = DataLoader(traindata, batch_size=15, num_workers = 7)
  validation_data = DataLoader(validdata, batch_size=15)
  validation_size = len(validation_data)
  """
  mean = torch.tensor([0.485, 0.456, 0.406])
  std = torch.tensor([0.229, 0.224, 0.225]) # the biggest value that can be normalized to is 2.64
  normalize = transforms.Normalize(mean.tolist(), std.tolist())
  
  
  print_every = 50
  save_every = 200
  i = iteration
  for epoch in range(1000):  # loop over the dataset multiple times

      running_lossD = 0.0
      running_lossG = 0.0
      train_loss = 0.0
      for data in dataset:
          i += 1
          # get the inputs; data is a list of [inputs, labels]
          inputs, real = data
          inputs = inputs.to(device)
          real = real.to(device)
          # zero the parameter gradients
          
          batch_size = len(inputs)
          
          disc.zero_grad()
          real_out = torch.sigmoid(disc(normalize(real))).mean()
          fakes = net(inputs)
          fake_out = torch.sigmoid(disc(normalize(fakes.detach())).view(-1)).mean()
          errD = 1-real_out + fake_out
          errD.backward()
          optimizer_disc.step()
          
          net.zero_grad()
          errG = 1 - torch.sigmoid(disc(normalize(fakes))).mean()
          errG.backward()
          optimizer.step()
          
          errorD = errD.item()
          errorG = errG.item()

          running_lossG += errorG
          running_lossD += errorD
          train_loss += errorG
          
          # print statistics
          if i % print_every == 0:
              print('[%d, %5d] lossG: %.4f' %
                    (epoch, i, running_lossG / print_every))
              print('[%d, %5d] lossD: %.4f' %
                    (epoch, i, running_lossD / print_every))
              running_lossD, running_lossG = 0.0, 0.0
          if i % save_every == save_every-1:
            train_losses.append(train_loss / save_every)
            print("Training loss:", train_loss / save_every)
            train_loss = 0.0
            iterations.append(i)
            saveNet(filename, net, optimizer, disc, optimizer_disc, iterations, train_losses, val_losses)
            print("Saved model!")
            
            with torch.no_grad():
              net.eval()
              percep_loss = 0
              pixel_loss = 0
              for inputs, labels in validation_data:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs_val = net(inputs)
                pix_loss = F.l1_loss(outputs_val, labels)
                per_loss = pix_loss #Temp
                percep_loss += per_loss.item()
                pixel_loss += pix_loss.item()

              percep_loss /= validation_size
              pixel_loss /= validation_size
              validation_loss = percep_loss + pixel_loss
              val_losses.append(validation_loss)
              
              print("Validation loss:", validation_loss, "Pixel:", pixel_loss, "Perceptual:", percep_loss)
              net.train()
              
                
