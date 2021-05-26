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
import torchvision
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
from torchvision.io.image import read_image, ImageReadMode

import common_parameters
from losses import VGG, perceptual_loss, sobel_filter, psnr, superHast, catmullHast

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
  torch.multiprocessing.freeze_support()
  torch.manual_seed(1337)
    
  print('cuda' if torch.cuda.is_available() else 'cpu')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  net = UNet(depth=5).to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=common_parameters.learning_rate)

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
  elif loss_str == "hast":
    criterion = lambda real, fake: F.l1_loss(real, fake) + 2*F.l1_loss(superHast(real, device), superHast(fake, device))
  elif loss_str == "hastCatmull":
    criterion = lambda real, fake: F.l1_loss(real, fake) + 4*F.l1_loss(catmullHast(real, device), catmullHast(fake, device))

  writer = SummaryWriter(common_parameters.relative_path + 'runs/' + filename.split('.')[0])
  print("Tensorboard saved at", common_parameters.relative_path + 'runs/' + filename.split('.')[0])

  filename = common_parameters.relative_path + filename

  iterations, train_losses, val_losses = loadNet(filename, net, optimizer, device)
  best_loss = min(val_losses) if len(val_losses) > 0 else 1e6
  print("Best validation loss:", best_loss)
  iteration = iterations[-1] if len(iterations) > 0 else -1

  #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-4, total_steps=common_parameters.end_iterations, cycle_momentum=False, last_epoch=iteration, div_factor=5, final_div_factor=1e1)

  net.train()
  net.to(device)
  
  batch_size = common_parameters.batch_size
  traindata = FolderSet(common_parameters.relative_path + "train")
  validdata = FolderSet(common_parameters.relative_path + "valid")

  dataset = DataLoader(traindata, batch_size=batch_size, num_workers = 4)
  validation_dataset = DataLoader(validdata, batch_size=16, num_workers = 4)
  
  validation_data = [i for i in validation_dataset]
  validation_size = len(validation_data)
  
  #dataset = DataLoader(FolderSet("text"), batch_size=10, num_workers = 7)

  print("Datasets loaded")
  print_every = 10
  save_every = 50
  i = iteration

  for epoch in range(1000):  # loop over the dataset multiple times

      running_loss = []
      train_loss = []
      for data in dataset:
          i += 1
          if i > common_parameters.end_iterations - 1:
            break
          # get the inputs; data is a list of [inputs, labels]
          inputs, real = data
          inputs = inputs.to(device)
          real = real.to(device)

          net.zero_grad()

          fakes = net(inputs)

          loss = criterion(real, fakes)
          loss.backward()
          optimizer.step()
          #scheduler.step()
          loss_item = loss.item()
          running_loss.append(loss_item)
          train_loss.append(loss_item)


          # print statistics
          if i % print_every == 0:
              print('[%d, %5d] loss: %.4f' %
                    (epoch, i, sum(running_loss)/len(running_loss)))
              writer.add_scalar("loss/train", sum(running_loss)/len(running_loss), i)
              running_loss = []
          if i % save_every == save_every-1:
            train_losses.append(sum(train_loss)/len(train_loss))
            iterations.append(i)
            train_loss = []
            saveNet(filename, net, optimizer, iterations, train_losses, val_losses)
            
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
              writer.add_scalar("loss/valid", validation_loss, i)
              writer.add_scalar("psnr/valid", psnr_score, i)

              speed_mini = read_image("speed-mini.png", mode=ImageReadMode.RGB).to(device).float() / 255.0
              writer.add_image("validation image", net(speed_mini.unsqueeze(0)).squeeze(), i)
              

              
              print("Validation loss:", validation_loss, "Mean PSNR:", psnr_score)#, "lr:", scheduler.get_last_lr())
              net.train()
              if validation_loss < best_loss:
                saveNet(filename + "_best", net, optimizer, iterations, train_losses, val_losses)
                print(f"New best loss: {best_loss} -> {validation_loss}")
                best_loss = validation_loss
            print("Saved model!")
      # This code makes sure that we break both loops if the inner loop is broken out of:
      else:
        continue
      break
  writer.close()
            
              
                
