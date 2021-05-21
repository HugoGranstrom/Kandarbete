import matplotlib.pyplot as plt
import torch
import time
import sys
import csv
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from net import *
from losses import *
from hqset import *
from unet import *

import common_parameters

if __name__ == '__main__':
  files = [
    ("original", "High-res"), ("downsampled", "Low-res"), ("bicubic", "Bicubic"), ("lanczos", "Lanczos"), ("net_l1", "L1"),
    ("net_l1GAN", "L1 + GAN"), ("net_mse", "MSE"), ("net_perceptualGAN", "Perceptual + GAN"),
    ("net_xtraGAN", "Sobel + Perceptual + GAN"), ("GANSobel", "Sobel + GAN"), 
    ("perceptual", "Perceptual"), ("sobel", "Sobel")
    ]
  rows = 3
  columns = 4
  foldername = input("Name of the folder (eg 0807): ")
  print("# files: ", len(files), "# slots: ", rows*columns)
  fig = plt.figure(figsize=(10, 8))
  plt.title("Comparison of models", y=1.08)
  plt.axis("off")
  for i, (filename, title) in enumerate(files):
    im = Image.open("Images/" + foldername + "/" + filename + ".png").convert("RGB")
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(im)
    plt.axis("off")
    plt.title(title)
  fig.tight_layout(w_pad=-9, h_pad=2)
  plt.savefig(foldername + ".png")
  plt.show()
  
