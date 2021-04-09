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


if __name__ == '__main__':
  torch.multiprocessing.freeze_support()

  csvfile = pd.read_csv("https://raw.githubusercontent.com/HugoGranstrom/Kandarbete/main/ImageUID.csv", names=["id"])
  ids = csvfile["id"].values
  print(len(ids))

  torch.manual_seed(1337)
    
  print('cuda' if torch.cuda.is_available() else 'cpu')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


  lr_min = 0.0001
  lr_max = 0.001

  net = UNet(depth=4, batch_norm=False).to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=lr_min)

  filename = "net_UNet_v2.pt"

  iteration, best_loss = loadNet(filename, net, optimizer, device)

  scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=500, last_epoch=iteration, mode="triangular", cycle_momentum=False)

  net.train()
  net.to(device)
  validation_size = 100
  vgg = Vgg16(requires_grad=False).to(device).eval()

  #dataset = OpenDataset(ids[:-validation_size], batch_size=15, SUPER_BATCHING=40, high_res_size=(256, 256), low_res_size=(128, 128))
  #validation_dataset = OpenDataset(ids[-validation_size:], batch_size=15, SUPER_BATCHING=1, high_res_size=(256, 256), low_res_size=(128, 128))
  
  traindata = FolderSet("train")
  validdata = FolderSet("valid")

  trainloader = DataLoader(traindata, batch_size=10, num_workers = 7)
  validloader = DataLoader(validdata, batch_size=8)
  validation_size = len(validdata)/8

  for epoch in range(1000):  # loop over the dataset multiple times

      running_loss = 0.0
      for i, data in enumerate(trainloader, iteration+1):
          #  lr_new = lr_min + (lr_max - lr_min) * math.exp(-i/7000)
          #  print("lr:", lr_new)
          #  for p in optimizer.param_groups:
          #    p['lr'] = lr_new
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs = inputs.to(device)
          labels = labels.to(device)
          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          
          #features_y = vgg(labels)
          #features_outputs = vgg(outputs)
          #loss = 0.5 * torch.mean((features_y.relu2_2 - features_outputs.relu2_2)**2)
          #loss += torch.mean(torch.abs(outputs - labels))
          #loss += torch.mean((outputs - labels) ** 2)
          loss = perceptual_loss(outputs, labels, vgg)
          loss += F.l1_loss(outputs, labels)
          loss.backward()
          optimizer.step()
          scheduler.step()
          running_loss += loss.item()
          
          # print statistics
          print_every = 10
          if i % print_every == print_every-1:
              print('[%d, %5d] loss: %.4f' %
                    (epoch + 1, i + 1, running_loss / print_every))
              running_loss = 0.0
          save_every = 80
          if i % save_every == save_every-1:
            with torch.no_grad():
              net.eval()
              percep_loss = 0
              pixel_loss = 0
              for inputs, labels in validloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs_val = net(inputs)
                #features_y = vgg(labels)
                #features_outputs = vgg(outputs)
                #per_loss = 0.5 * torch.mean((features_y.relu2_2 - features_outputs.relu2_2)**2)
                per_loss = perceptual_loss(outputs_val, labels, vgg)
                pix_loss = F.l1_loss(outputs_val, labels)
                percep_loss += per_loss.item()
                pixel_loss += pix_loss.item()

              percep_loss /= validation_size
              pixel_loss /= validation_size
              validation_loss = percep_loss + pixel_loss
              
              print("Validation loss:", validation_loss, "Pixel:", pixel_loss, "Perceptual:", percep_loss, "lr:", scheduler.get_last_lr())
              net.train()
              if True:
                best_loss = validation_loss
                saveNet(filename, net, optimizer, i+1, best_loss)
                print(f"Saving model, new best loss: {best_loss} -> {validation_loss}")
              
                
