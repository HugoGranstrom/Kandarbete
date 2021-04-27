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

class AdverserialModel(nn.Module):
  def __init__(this, high_res):
    super().__init__()
    this.model = nn.Sequential(
      nn.Conv2d(3, 16, 3,padding=1), # 3*3*3*16 = 432
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(16, 32, 3,padding=1, stride=2), # 4608
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(32, 64, 3,padding=1, stride=2), # 18 432
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(64, 128, 3,padding=1, stride=2), # 73 728
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(128, 128, 3,padding=1, stride=2), # 147 456
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(128, 128, 3,padding=1, stride=2), # 147 456
      nn.LeakyReLU(0.2, inplace=True),
      
      nn.Flatten(),
      
      nn.Linear(int(128*high_res*high_res/1024), 8096), # 8 388 608
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(8096, 1024),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(1024, 128),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(128, 1)
    )

  def forward(this, x):
    return this.model(x)

def sobel_filter(y, device):
  kernel_x = torch.tensor([[1, 0, -1],[2,0,-2],[1,0,-1]]).view(1,1,3,3).expand(3,-1,-1,-1).float().to(device)
  kernel_y = torch.tensor([[1, 2, 1],[0,0,0],[-1,-2,-1]]).view(1,1,3,3).expand(3,-1,-1,-1).float().to(device)
  Gx = F.conv2d(y, kernel_x, groups=y.shape[1])
  Gy = F.conv2d(y, kernel_y, groups=y.shape[1])
  return (Gx**2 + Gy**2 + 1e-8).sqrt()

if __name__ == '__main__':
  torch.multiprocessing.freeze_support()

  csvfile = pd.read_csv("ImageUID.csv", names=["id"])
  ids = csvfile["id"].values
  print(len(ids))

  torch.manual_seed(1337)
    
  print('cuda' if torch.cuda.is_available() else 'cpu')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  lr_min = 0.0001
  lr_max = 0.0005

  net = UNet(depth=5).to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)
  
  disc = AdverserialModel(256).to(device)

  optimizer_disc = torch.optim.Adam(disc.parameters(), lr=0.0002)

  filename = "GAN_UNet_v1.pt"

  iterations, train_losses, val_losses = loadNet(filename, net, optimizer, disc, optimizer_disc, device)
  best_loss = min(val_losses) if len(val_losses) > 0 else 1e6
  print("Best validation loss:", best_loss)
  iteration = iterations[-1] if len(iterations) > 0 else -1
  #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max, step_size_up=2000, last_epoch=iteration, mode="triangular", cycle_momentum=False)

  net.train()
  net.to(device)
  validation_size = 100
  """dataset = OpenDataset(ids[:-validation_size], batch_size=batch_size, SUPER_BATCHING=40, high_res_size=(256, 256), low_res_size=(128, 128))
  batch_size = 8

  dataset = OpenDataset(ids[:-validation_size], batch_size=batch_size, SUPER_BATCHING=40, high_res_size=(256, 256), low_res_size=(128, 128))
  validation_dataset = OpenDataset(ids[-validation_size:], batch_size=batch_size, SUPER_BATCHING=1, high_res_size=(256, 256), low_res_size=(128, 128))
  validation_data = [i for i in validation_dataset]
  validation_size = len(validation_data)
  """
  traindata = FolderSet("train")
  validdata = FolderSet("valid")

  dataset = DataLoader(traindata, batch_size=8, num_workers = 7)
  validation_data = DataLoader(validdata, batch_size=10)
  validation_size = len(validation_data)
  
  print_every = 50
  save_every = 500
  disc_training_factor = 5
  i = iteration
  
  criterion = nn.BCEWithLogitsLoss()
  for epoch in range(1000):  # loop over the dataset multiple times

      running_lossD, running_lossG, running_loss = 0.0, 0.0, 0.0
      train_loss = 0.0
      for data in dataset:
          i += 1
          # get the inputs; data is a list of [inputs, labels]
          inputs, real = data
          inputs = inputs.to(device)
          real = real.to(device)
          
          batch_size = len(inputs)
          
          fake_labels = torch.zeros(batch_size).unsqueeze(-1).to(device)
          real_labels = torch.ones(batch_size).unsqueeze(-1).to(device)
          
          disc.zero_grad()
          real_out = criterion(disc(real),real_labels)
          fakes = net(inputs)
          fake_out = criterion(disc(fakes.detach()),fake_labels)
          errD = real_out + fake_out
          errD.backward()
          optimizer_disc.step()
          
          running_lossD += errD.item()
          
          if i % disc_training_factor == 0:
            net.zero_grad()
            errG = criterion(disc(fakes), fake_labels)
            loss = 0.001*errG + F.l1_loss(fakes,real) + F.l1_loss(sobel_filter(fakes,device),sobel_filter(real,device))
            loss.backward()
            optimizer.step()
            running_lossG += errG.item()
            running_loss += loss.item()

          #loss = perceptual_loss(outputs, real, vgg)
          #loss += F.l1_loss(outputs, real)
          #loss.backward()
          #optimizer.step()
          #scheduler.step()
          # print statistics
          if i % print_every == 0:
              print('[%d, %5d, (%d)] loss: %.4f' %
                    (epoch, i, i/disc_training_factor, running_loss / (print_every/disc_training_factor)))
              print('[%d, %5d] lossG: %.4f' %
                    (epoch, i, running_lossG / (print_every/disc_training_factor)))
              print('[%d, %5d] lossD: %.4f' %
                    (epoch, i, running_lossD / print_every))
              running_lossD, running_lossG, running_loss = 0.0, 0.0, 0.0
          if i % save_every == save_every-1:
            train_loss = 0.0
            iterations.append(i)
            saveNet(filename, net, optimizer, disc, optimizer_disc, iterations, train_losses, val_losses)
            print("Saved model!")
            """
            with torch.no_grad():
              net.eval()
              percep_loss = 0
              pixel_loss = 0
              for inputs, labels in validation_data:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs_val = net(inputs)
                per_loss = perceptual_loss(outputs_val, labels, vgg)
                pix_loss = F.l1_loss(outputs_val, labels)
                percep_loss += per_loss.item()
                pixel_loss += pix_loss.item()

              percep_loss /= validation_size
              pixel_loss /= validation_size
              validation_loss = percep_loss + pixel_loss
              val_losses.append(validation_loss)
              
              print("Validation loss:", validation_loss, "Pixel:", pixel_loss, "Perceptual:", percep_loss, "lr:", scheduler.get_last_lr())
              net.train()
              if validation_loss < best_loss:
                saveNet(filename + "_best", net, optimizer, disc, optimizer_disc, iterations, train_losses, val_losses)
                print(f"New best loss: {best_loss} -> {validation_loss}")
                best_loss = validation_loss
            """
              
                
