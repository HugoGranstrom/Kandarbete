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
from torchvision.io.image import read_image, ImageReadMode

import common_parameters
from losses import VGG, perceptual_loss, sobel_filter, psnr, AdverserialModel

from torch.utils.tensorboard import SummaryWriter

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
  elif loss_str == "xtra-allt":
    vgg = VGG().eval().to(device)
    criterion = lambda real, fake: F.l1_loss(real, fake) + F.l1_loss(sobel_filter(real, device), sobel_filter(fake, device)) + 0.2*perceptual_loss(real, fake, vgg)
  else:
    raise RuntimeError(loss_str + " is not a valid loss")

  writer = SummaryWriter(common_parameters.relative_path + 'runs/' + filename.split('.')[0])
  filename = common_parameters.relative_path + filename


  iterations, train_losses, val_losses = loadNetGAN(filename, net, optimizer, disc, optimizer_disc, device)
  best_loss = min(val_losses) if len(val_losses) > 0 else 1e6
  print("Best validation loss:", best_loss)
  iteration = iterations[-1] if len(iterations) > 0 else -1
  

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
  print_every = 50
  save_every = 500
  i = iteration
  
  speed_mini = read_image("speed-mini.png", mode=ImageReadMode.RGB).to(device).float() / 255.0
  
  for epoch in range(1000):  # loop over the dataset multiple times

      running_lossD, running_lossG, running_loss = [],[],[]
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
          
          net.zero_grad()
          
          real_out = disc(real)
          fakes = net(inputs)
          
          fake_out = disc(fakes)
          errG = (torch.mean((real_out - torch.mean(fake_out) + 1)**2) + torch.mean((fake_out - torch.mean(real_out) - 1)**2))/2
          
          loss = 0.001*errG + criterion(real, fakes)
          loss.backward(retain_graph=True)
          optimizer.step()

          disc.zero_grad()
          fake_out = disc(fakes.detach())
          errD = (torch.mean((real_out - torch.mean(fake_out) - 1)**2) + torch.mean((fake_out - torch.mean(real_out) + 1)**2))/2
          
          errD.backward()
          running_lossD.append(errD.item())
          
          optimizer_disc.step()


          running_lossG.append(errG.item())
          loss_item = loss.item()
          running_loss.append(loss_item)
          train_loss += loss_item


          # print statistics
          if i % print_every == 0:
            print('[%d, %5d] loss: %.4f' %
                  (epoch, i, sum(running_loss)/len(running_loss)))
            print('[%d, %5d] lossG: %.4f' %
                  (epoch, i, sum(running_lossG)/len(running_lossG)))
            print('[%d, %5d] lossD: %.4f' %
                  (epoch, i, sum(running_lossD)/len(running_lossD)))
            writer.add_scalar("loss/train", sum(running_loss)/len(running_loss), i)
            writer.add_scalar("loss/train_generator", sum(running_lossG)/len(running_lossG), i)
            writer.add_scalar("loss/train_discriminator", sum(running_lossD)/len(running_lossD), i)
            with torch.no_grad():
              net.eval()
              writer.add_image("train image", net(speed_mini.unsqueeze(0)).squeeze(), i)
            net.train()
            running_lossD, running_lossG, running_loss = [],[],[]
          if i % save_every == save_every-1:
            train_losses.append(train_loss/save_every)
            iterations.append(i)
            train_loss = 0.0
            saveNetGAN(filename, net, optimizer, disc, optimizer_disc, iterations, train_losses, val_losses)
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

              writer.add_image("validation image", net(speed_mini.unsqueeze(0)).squeeze(), i)
              
              print("Validation loss:", validation_loss, "Mean PSNR:", psnr_score)
              net.train()
              if validation_loss < best_loss:
                saveNetGAN(filename + "_best", net, optimizer, disc, optimizer_disc, iterations, train_losses, val_losses)
                print(f"New best loss: {best_loss} -> {validation_loss}")
                best_loss = validation_loss
            print("Saved model!")
      # This code makes sure that we break both loops if the inner loop is broken out of:
      else:
        continue
      break
  writer.close() 
              
                
