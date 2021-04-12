from concurrent import futures
import os
import re
import sys

import boto3
import botocore
import tqdm

import torch.nn as nn
import torch.nn.functional as F

import threading

import torch
from torchvision import transforms
from PIL import Image
import time
from multiprocessing import Process, Queue
import os
import numpy as np

import pandas as pd

import random

def download_one_image(bucket, split, image_id, download_folder):
  try:
    bucket.download_file(f'{split}/{image_id}.jpg',
                         os.path.join(download_folder, f'{image_id}.jpg'))
  except botocore.exceptions.ClientError as exception:
    sys.exit(
        f'ERROR when downloading image `{split}/{image_id}`: {str(exception)}')

class NinetiesRotation:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

class OpenDataset:
  def __init__(this, ids, batch_size,SUPER_BATCHING = 30, high_res_size = (200, 200), low_res_size = (100, 100)):
    ## ids: list of Google Open Image Dataset ids
    ## batch_size: number of images in each batch

    this.bucket = boto3.resource(
      's3', config=botocore.config.Config(
          signature_version=botocore.UNSIGNED)).Bucket('open-images-dataset')

    this.ids = np.array(ids)
    this.batch_size = batch_size
    this.running = True

    this.current_batch = 0
    this.batch_queue = Queue()
    this.SUPER_BATCHING = SUPER_BATCHING
    this.current_epoch = 0
    this.high_res_size = high_res_size
    this.low_res_size = low_res_size
    # Crops the images and adds padding if needed:
    this.crop_transform = transforms.Compose([
                                              transforms.RandomCrop(high_res_size, padding=None, pad_if_needed=True),
                                              transforms.RandomHorizontalFlip(),
                                              NinetiesRotation()
                                              ])
    # Transforms a high-res image to a downscaled low-res image
    this.X_transforms = transforms.Compose([
                                            transforms.Resize(low_res_size, transforms.InterpolationMode.BILINEAR)
                                            ])
    this.toTensor = transforms.Compose([transforms.ToTensor()])
    try:
      os.mkdir("imgs")
    except OSError:
      pass # Directory already there

  def __iter__(this):
    this.current_batch = 0
    this.current_epoch = 0
    np.random.shuffle(this.ids) # shuffle ids'
    this.batch_thread_running = True
    this.batch_thread = threading.Thread(target=this._batch_process)
    this.batch_thread.start()
    return this

  def __next__(this):
    ## load next batch (X, Y)
    Xs = []
    Ys = []

    while this.batch_queue.empty():
      time.sleep(0.001)
    batch_data = this.batch_queue.get()
		
    if batch_data == False:
      this.batch_thread_running = False
      raise StopIteration
    
    # Return as (X_batch, Y_batch) where X_batch and Y_batch are two Tensors
    return batch_data

  def download_image(this, id):
    #print(f"Downloding {id}")
    if not os.path.isfile(f"imgs/{id}.jpg"): # Check if already downloaded.
      download_one_image(this.bucket,"train",id,"imgs")
    im = Image.open(f"imgs/{id}.jpg")
    im = im if im.mode == "HSV" else im.convert("HSV")
    return im

  def _batch_process(this):
    while this.batch_thread_running:
      while this.batch_queue.qsize() > 200:
        time.sleep(0.005)
      this.prepare_one_batch()

  def prepare_one_batch(this):
    if this.batch_size*(this.current_batch+1) > len(this.ids):
      this.batch_queue.put(False)
      return
    batch_imgids = this.ids[this.current_batch*this.batch_size:this.batch_size*(this.current_batch+1)]
    this.current_batch += 1

    SUPER_BATCHING = this.SUPER_BATCHING

    Xss = [[] for i in range(SUPER_BATCHING)]
    Yss = [[] for i in range(SUPER_BATCHING)]
    #print(len(batch_imgids))
    for id in batch_imgids:
      im = this.download_image(id)
      # Apply this.crop_transform first to get Y
      # Apply this.X_transforms on Y to get X
      for i in range(SUPER_BATCHING):
        transformed_im = this.crop_transform(im)
        Yss[i].append(this.toTensor(transformed_im))
        Xss[i].append(this.toTensor(this.X_transforms(transformed_im)))

    for i in range(SUPER_BATCHING):
      batch = (torch.stack(Xss[i]), torch.stack(Yss[i]));
      this.batch_queue.put(batch)