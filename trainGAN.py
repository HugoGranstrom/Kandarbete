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

import common_parameters
from losses import VGG, perceptual_loss, sobel_filter, psnr, AdverserialModel


if __name__ == '__main__':
  torch.multiprocessing.freeze_support()

  torch.manual_seed(1337)
    
  print('cuda' if torch.cuda.is_available() else 'cpu')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  net = UNet(depth=5).to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=common_parameters.learning_rate)
  
  disc = AdverserialModel(256).to(device)

  optimizer_disc = torch.optim.Adam(disc.parameters(), lr=common_parameters.learning_rate)

  if len(sys.argv) != 3: raise RuntimeError("Two command-line arguments must be given, the model's filename and the type of loss")
  filename = sys.argv[1]
  loss_str = sys.argv[2]
  # criterion is a function that takes the arguments (real_imgs, fake_imgs) in that order!
  if loss_str == "mse":
    criterion = F.mse_loss
  elif loss_str == "l1":
    criterion = F.l1_loss
  elif loss_str == "sobel":
    criterion = lambda real, fake: F.l1_loss(real, fake) + F.l1_loss(sobel_filter(real, device), sobel_filter(fake, device))
  elif loss_str == "perceptual":
    vgg = VGG().eval().to(device)
    criterion = lambda real, fake: F.l1_loss(real, fake) + perceptual_loss(real, fake, vgg)

  iterations, train_losses, val_losses = loadNetGAN(filename, net, optimizer, disc, optimizer_disc, device)
  best_loss = min(val_losses) if len(val_losses) > 0 else 1e6
  print("Best validation loss:", best_loss)
  iteration = iterations[-1] if len(iterations) > 0 else -1
  
  net.train()
  net.to(device)
  

  batch_size = common_parameters.batch_size

  traindata = FolderSet("train")
  validdata = FolderSet("valid")

  dataset = DataLoader(traindata, batch_size=batch_size, num_workers = 4)
  validation_dataset = DataLoader(validdata, batch_size=16, num_workers = 4)
  
  validation_data = [i for i in validation_dataset]
  validation_size = len(validation_data)
  
  #dataset = DataLoader(FolderSet("text"), batch_size=10, num_workers = 7)
  
  print("Datasets loaded")
  print_every = 100
  save_every = 500
  i = iteration
  
  for epoch in range(1000):  # loop over the dataset multiple times

      running_lossD, running_lossG, running_loss = 0.0, 0.0, 0.0
      train_loss = 0.0
      for data in dataset:
          i += 1
          if i > common_parameters.end_iterations - 1:
            break
          # get the inputs; data is a list of [inputs, labels]
          inputs, real = data
          inputs = inputs.to(device)
          real = real.to(device)
          
          batch_size = len(inputs)
          
          #real_labels = torch.ones(batch_size).unsqueeze(-1).to(device)
          
          # Freeze weights of disc while training generator:
          for param in disc.parameters():
            param.requires_grad = False

          real_out = disc(real)
          fakes = net(inputs)
          fake_out = disc(fakes)

          net.zero_grad()
          errG = (torch.mean((real_out - torch.mean(fake_out) + 1)**2) + torch.mean((fake_out - torch.mean(real_out) - 1)**2))/2
          
          loss = 0.001*errG + criterion(real, fakes)
          loss.backward(retain_graph=True)
          optimizer.step()
          
          # Unfreeze disc's weights
          for param in disc.parameters():
            param.requires_grad = True

          disc.zero_grad()
          fake_out = disc(fakes.detach())
          errD = (torch.mean((real_out - torch.mean(fake_out) - 1)**2) + torch.mean((fake_out - torch.mean(real_out) + 1)**2))/2
          
          errD.backward()
          running_lossD += errD.item()
          
          optimizer_disc.step()


          running_lossG += errG.item()
          loss_item = loss.item()
          running_loss += loss_item
          train_loss += loss_item


          # print statistics
          if i % print_every == 0:
              print('[%d, %5d] loss: %.4f' %
                    (epoch, i, running_loss / print_every))
              print('[%d, %5d] lossG: %.4f' %
                    (epoch, i, running_lossG / print_every))
              print('[%d, %5d] lossD: %.4f' %
                    (epoch, i, running_lossD / print_every))
              running_lossD, running_lossG, running_loss = 0.0, 0.0, 0.0
          if i % save_every == save_every-1:
            train_losses.append(train_loss/save_every)
            train_loss = 0.0
            iterations.append(i)
            saveNetGAN(filename, net, optimizer, disc, optimizer_disc, iterations, train_losses, val_losses)
            print("Saved model!")
            with torch.no_grad():
              net.eval()
              criterion_loss = 0.0
              psnr_score = 0
              for inputs, labels in validation_data:
                inputs = inputs.to(device)
                real_val = labels.to(device)
                fakes_val = net(inputs)
                criterion_loss += criterion(real_val, fakes_val).item()
                psnr_score += psnr(real_val, fakes_val).item()
                
              criterion_loss /= validation_size
              psnr_score /= validation_size
              validation_loss = criterion_loss
              val_losses.append(validation_loss)
              
              print("Validation loss:", validation_loss, "Mean PSNR:", psnr_score)
              net.train()
              if validation_loss < best_loss:
                saveNetGAN(filename + "_best", net, optimizer, disc, optimizer_disc, iterations, train_losses, val_losses)
                print(f"New best loss: {best_loss} -> {validation_loss}")
                best_loss = validation_loss
      # This code makes sure that we break both loops if the inner loop is broken out of:
      else:
        continue
      break      
              
                
