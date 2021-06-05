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
import random
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
    def __init__(self, low_res_size):
        self.low_res_size = low_res_size
        self.methods = [transforms.InterpolationMode.LANCZOS, transforms.InterpolationMode.BICUBIC, transforms.InterpolationMode.BILINEAR]

    def __call__(self, x):
        mode = random.choice(self.methods)
        return transforms.functional.resize(x, self.low_res_size, mode)

class FolderSet(Dataset):
  def __init__(this, root_dir, high_res_size = (256, 256), low_res_size = (128, 128), center=False):
    this.high_res_size = high_res_size
    this.low_res_size = low_res_size
    if center:
      this.crop_transform = transforms.CenterCrop(high_res_size)
    else:
      this.crop_transform = transforms.Compose([
                                              transforms.RandomCrop(high_res_size, padding=None, pad_if_needed=True),
                                              transforms.RandomHorizontalFlip(),
                                              NinetiesRotation()
                                              ])
    # Transforms a high-res image to a downscaled low-res image
    this.X_transforms = transforms.Compose([
                                            #transforms.GaussianBlur(3), #Simulate camera point
                                            transforms.Resize(low_res_size, transforms.InterpolationMode.BILINEAR)
                                            #RandomDownsampling(low_res_size)
                                            ])
    this.toTensor = transforms.Compose([transforms.ToTensor()])
    
    this.files = glob.glob(f"{root_dir}/*.png")
    if not center:
      random.shuffle(this.files)
    this.length = len(this.files)
      
  def __len__(this):
    return this.length
    
  def __getitem__(this,idx):
    im = Image.open(this.files[idx])
    im = im if im.mode == "RGB" else im.convert("RGB")
    
    transformed_im = this.crop_transform(im)
    Ys = this.toTensor(transformed_im)
    Xs = this.toTensor(this.X_transforms(transformed_im))
    
    return (Xs, Ys)
    
class FolderSetFull(Dataset):
  def __init__(this, root_dir):
    this.toTensor = transforms.Compose([transforms.ToTensor()])
    
    this.files = glob.glob(f"{root_dir}/*.png")
    this.length = len(this.files)
      
  def __len__(this):
    return this.length
    
  def __getitem__(this,idx):
    image = Image.open(this.files[idx])
    image = image if image.mode == "RGB" else image.convert("RGB")
    im, padding, original_width, original_height = this.compat_pad(image, 5)
    Ys = this.toTensor(im)
    sz = im.size
    Xs = this.toTensor(transforms.Resize(((int)(sz[1]/2),(int)(sz[0]/2)), transforms.InterpolationMode.LANCZOS)(im))
    
    return (Xs, Ys)
    
  def compat_pad(this, image, network_depth):
    n = 2**network_depth
    if isinstance(image, Image.Image):
      width, height = image.size
    elif isinstance(image, torch.Tensor):
      shape = image.shape
      height, width = shape[1], shape[2]
    else:
      raise ValueError("image wasn't a PIL image or a Pytorch Tensor")
    pad_width = n - width % n
    if pad_width == n: pad_width = 0
    pad_height = n - height % n
    if pad_height == n: pad_height = 0
    if pad_width % 2 == 0:
      pad_left, pad_right = pad_width//2, pad_width//2
    else:
      pad_left, pad_right = pad_width//2, pad_width//2 + 1
    if pad_height % 2 == 0:
      pad_up, pad_down = pad_height//2, pad_height//2
    else:
      pad_up, pad_down = pad_height//2, pad_height//2 + 1
    padding = [pad_left, pad_up, pad_right, pad_down]
    padded_im = transforms.Pad(padding)(image)
    return padded_im, padding, width, height