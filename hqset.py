from concurrent import futures
import os
import glob
import torch.nn as nn
import torch.nn.functional as F

import threading

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import time
import os
import numpy as np

import pandas as pd

import random

class NinetiesRotation:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)
        
class RandomDownsampling:
    """Rotate by one of the given angles."""

    def __init__(self, low_res_size):
        self.low_res_size = low_res_size
        self.methods = [transforms.InterpolationMode.LANCZOS, transforms.InterpolationMode.BICUBIC, transforms.InterpolationMode.BILINEAR]

    def __call__(self, x):
        mode = random.choice(self.methods)
        return transforms.functional.resize(x, self.low_res_size, mode)


class FolderSet(Dataset):
  def __init__(this, root_dir, high_res_size = (256, 256), low_res_size = (128, 128), length_multiplier = 8):
    this.high_res_size = high_res_size
    this.low_res_size = low_res_size
    this.crop_transform = transforms.Compose([
                                              transforms.RandomCrop(high_res_size, padding=None, pad_if_needed=True),
                                              transforms.RandomHorizontalFlip(),
                                              NinetiesRotation()
                                              ])
    # Transforms a high-res image to a downscaled low-res image
    this.X_transforms = transforms.Compose([
                                            RandomDownsampling()
                                            ])
    this.toTensor = transforms.Compose([transforms.ToTensor()])
    
    this.files = glob.glob(f"{root_dir}/*.png")
    this.length = len(this.files)*length_multiplier
      
  def __len__(this):
    return this.length
    
  def __getitem__(this,idx):
    im = Image.open(this.files[random.randrange(0,len(this.files))])
    im = im if im.mode == "RGB" else im.convert("RGB")
    
    transformed_im = this.crop_transform(im)
    Ys = this.toTensor(transformed_im)
    Xs = this.toTensor(this.X_transforms(transformed_im))
    
    return (Xs, Ys)