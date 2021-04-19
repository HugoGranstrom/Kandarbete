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
      nn.Linear(int(128*high_res*high_res/1024), 1024), # 8 388 608
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(1024, 128),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(128, 1),
      nn.Sigmoid()
    )

  def forward(this, x):
    return this.model(x)

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
  """dataset = OpenDataset(ids[:-validation_size], batch_size=batch_size, SUPER_BATCHING=40, high_res_size=(256, 256), low_res_size=(128, 128))
  validation_dataset = OpenDataset(ids[-validation_size:], batch_size=batch_size, SUPER_BATCHING=1, high_res_size=(256, 256), low_res_size=(128, 128))
  validation_data = [i for i in validation_dataset]
  validation_size = len(validation_data)
  """
  traindata = FolderSet("train")
  validdata = FolderSet("valid")

  dataset = DataLoader(traindata, batch_size=15, num_workers = 7)
  validation_data = DataLoader(validdata, batch_size=15)
  validation_size = len(validation_data)
  
  print_every = 50
  save_every = 1
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
          optimizer_disc.zero_grad()
          
          batch_size = len(inputs)
          
          real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
          output = disc(real)
          errD_real = criterion(output.squeeze(), real_labels.squeeze())
          errD_real.backward()

          # forward + backward + optimize
          fakes = net(inputs)
          fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
          output = disc(fakes.detach()).view(-1)
          errD_fake = criterion(output.squeeze(), fake_labels.squeeze())
          errD_fake.backward()
          errD = errD_real + errD_fake
          optimizer_disc.step()

          optimizer.zero_grad()
          output = disc(fakes).view(-1)
          errG = criterion(output, real_labels)
          errG.backward()
          optimizer.step()

          errorD = errD.mean().item()
          errorG = errG.mean().item()
          #loss = perceptual_loss(outputs, real, vgg)
          #loss += F.l1_loss(outputs, real)
          #loss.backward()
          #optimizer.step()
          #scheduler.step()
          running_lossG += errorG
          running_lossD += errorD
          train_loss += errorG + errorD
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
              
                
