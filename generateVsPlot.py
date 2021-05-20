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
  filePairs = [
    (("net_l1", "L1"), ("net_l1GAN", "L1 + GAN")),
    (("net_mse", "MSE"), ("sobel", "Sobel")), 
    (("net_perceptualGAN", "Perceptual + GAN"), ("perceptual", "Perceptual")),
    (("net_xtraGAN", "Sobel + Perceptual + GAN"), ("GANSobel", "Sobel + GAN")), 
    (("net_l1GAN", "L1 + GAN"), ("net_mse", "MSE"))
    ]
  ("original", "High-res"), ("lanczos", "Lanczos")
  rows = 2
  columns = 2
  foldername = input("Name of the folder (eg 0807): ")
  lanczos_im = Image.open("Images/" + foldername + "/lanczos.png")
  original_im = Image.open("Images/" + foldername + "/original.png")
  for i, filePair in enumerate(filePairs):
    im1 = Image.open("Images/" + foldername + "/" + filePair[0][0] + ".png").convert("RGB")
    im2 = Image.open("Images/" + foldername + "/" + filePair[1][0] + ".png").convert("RGB")
    title1 = filePair[0][1]
    title2 = filePair[1][1]
    fig = plt.figure(figsize=(10, 10))
    plt.title("Comparasion of models", y=1.08)
    plt.axis("off")
    fig.add_subplot(rows, columns, 1)
    plt.imshow(original_im)
    plt.axis("off")
    plt.title("High-res")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(lanczos_im)
    plt.axis("off")
    plt.title("Lanczos")
    fig.add_subplot(rows, columns, 3)
    plt.imshow(im1)
    plt.axis("off")
    plt.title(title1)
    fig.add_subplot(rows, columns, 4)
    plt.imshow(im2)
    plt.axis("off")
    plt.title(title2)
    fig.tight_layout(w_pad=-9, h_pad=2)
    plt.savefig(foldername + "-" + str(i) + ".png")
    #plt.show()
  
