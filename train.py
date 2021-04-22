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

from collections import namedtuple

import torch
from torchvision import models

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        #for x in range(16, 23):
        #    self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

def perceptual_loss(y, y_hat, vgg):
  """Normalizes y and y_hat, runs them through vgg and compares intermediate layers and returns the perceptual loss"""
  mean = torch.tensor([0.485, 0.456, 0.406])
  std = torch.tensor([0.229, 0.224, 0.225]) # the biggest value that can be normalized to is 2.64
  normalize = transforms.Normalize(mean.tolist(), std.tolist())
  unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

  features_y = vgg(normalize(y))
  features_y_hat = vgg(normalize(y_hat))
  loss = 0.5 * F.mse_loss(features_y_hat.relu2_2, features_y.relu2_2)
  return loss

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

  scale_power = 1
  n_blocks = 1

  net = UNet(depth=5, scale_power=scale_power, n_blocks=n_blocks).to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=lr_min)

  filename = "net_UNet.pt"

  iterations, train_losses, val_losses = loadNet(filename, net, optimizer, device)
  best_loss = min(val_losses) if len(val_losses) > 0 else 1e6
  print("Best validation loss:", best_loss)
  iteration = iterations[-1] if len(iterations) > 0 else -1
  scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max, step_size_up=2000, last_epoch=iteration, mode="triangular", cycle_momentum=False)

  net.train()
  net.to(device)
  validation_size = 100
  vgg = Vgg16(requires_grad=False).to(device).eval()

  high_res_size = (256, 256)
  low_res_size = (high_res_size[0] // 2**scale_power, high_res_size[1] // 2**scale_power)
  print("Highres:", high_res_size, "Lowres:", low_res_size)

  batch_size = 4
  dataset = OpenDataset(ids[:-validation_size], batch_size=batch_size, SUPER_BATCHING=40, high_res_size=high_res_size, low_res_size=low_res_size)
  validation_dataset = OpenDataset(ids[-validation_size:], batch_size=batch_size, SUPER_BATCHING=1, high_res_size=high_res_size, low_res_size=low_res_size)
  validation_data = [i for i in validation_dataset]
  validation_size = len(validation_data)
  """
  traindata = FolderSet("train")
  validdata = FolderSet("valid",length_multiplier = 1)

  dataset = DataLoader(traindata, batch_size=20, num_workers = 7)
  validation_data = DataLoader(validdata, batch_size=25)
  validation_size = len(validation_data)
  """
  
  print_every = 100
  save_every = 500
  i = iteration
  for epoch in range(1000):  # loop over the dataset multiple times

      sobel_running_loss = 0.0
      pix_running_loss = 0.0
      running_loss = 0.0
      train_loss = 0.0
      for data in dataset:
          i += 1
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs = inputs.to(device)
          labels = labels.to(device)
          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          
          sobel_loss = 0.1*F.mse_loss(sobel_filter(outputs,device), sobel_filter(labels.detach(),device))
          pixel_loss = F.mse_loss(outputs, labels)
          
          sobel_running_loss += sobel_loss.item()
          pix_running_loss += pixel_loss.item()
          
          loss = sobel_loss
          loss += pixel_loss
          
          
          loss.backward()
          optimizer.step()
          scheduler.step()
          
          running_loss += loss.item()
          train_loss += loss.item()
          # print statistics
          if i % print_every == print_every-1:
              print('[%d, %d] loss: %.4f (Sobel: %.4f, Pixel: %.4f)' %
                    (epoch, i, running_loss / print_every, sobel_running_loss / print_every, pix_running_loss / print_every))
              running_loss = 0.0
              sobel_running_loss = 0.0
              pix_running_loss = 0.0
          if i % save_every == save_every-1:
            train_losses.append(train_loss / save_every)
            print("Training loss:", train_loss / save_every)
            train_loss = 0.0
            iterations.append(i)
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
                saveNet(filename + "_best", net, optimizer, iterations, train_losses, val_losses)
                print(f"New best loss: {best_loss} -> {validation_loss}")
                best_loss = validation_loss
              saveNet(filename, net, optimizer, iterations, train_losses, val_losses)
              print("Saved model!")
              
                
