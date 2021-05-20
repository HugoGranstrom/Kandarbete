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

def compat_pad(image, network_depth):
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

if __name__ == '__main__':
  filenames = input("Enter model files: ").split(" ")
  for filename_raw in filenames:
    filename = common_parameters.relative_path + filename_raw
    modelname = filename_raw.split(".")[0]
    imgs = ["0807.png", "0869.png", "0898.png", "0839.png", "0885.png"]
    crops = [(926, 704, 926+256, 704+256), (926, 704, 926+256, 704+256), (1600,600,1600+256, 600+256), (750,130,750+256, 130+256), (1550,350,1550+256, 350+256)]
    device_name = "none"
    if torch.cuda.is_available():
      while device_name != 'cuda' and device_name != 'cpu':
        device_name = input("Enter device ('cuda', 'cpu'):")
        if device_name == "":
          device_name = 'cuda'
          print('cuda')
    else:
      device_name = "cpu"
      print('cpu')

    device = torch.device(device_name)
    
    net = UNet(depth=5)
    loadNetEval(filename, net, device)
    net.to(device)
    net.eval()
    
    files = [common_parameters.relative_path + "valid/" + file for file in imgs]
    toTensor = transforms.Compose([transforms.ToTensor()])
    
    for i in range(len(files)):
      image = Image.open(files[i]).crop(crops[i])
      image = image if image.mode == "RGB" else image.convert("RGB")
      real = toTensor(image).unsqueeze(0)
      
      sz = image.size
      resized_im = transforms.Resize((sz[1]//2,sz[0]//2), transforms.InterpolationMode.BILINEAR)(image)
      im, padding, original_width, original_height = compat_pad(resized_im, 5)
      inputs = toTensor(im).unsqueeze(0)
      inputs = inputs.to(device)
      real = real.to(device)
      
      with torch.no_grad():
        y = net(inputs)

      #in_crop = transforms.functional.crop(inputs.squeeze(), padding[1]//2, padding[0]//2, original_height//2, original_width//2)
      im2 = resized_im
      rz_size = (im2.size[1]*2, im2.size[0]*2)
      y_lanz = transforms.Resize(rz_size, transforms.InterpolationMode.LANCZOS)(im2)
      y_bicub = transforms.Resize(rz_size, transforms.InterpolationMode.BICUBIC)(im2)
      y_net = transforms.ToPILImage()(y.squeeze())
      y_net = transforms.functional.crop(y_net, 2*padding[1], 2*padding[0], 2*original_height, 2*original_width)

      im_dir = common_parameters.relative_path + "Images/"
      save_dir = im_dir + imgs[i].split(".")[0] + "/" # Images/0807/ for example
      if not os.path.exists(im_dir):
        os.mkdir(im_dir)
      print("Saving in:", save_dir)
      if not os.path.exists(save_dir):
        os.mkdir(save_dir)
      y_lanz.save(save_dir + "lanczos.png")
      y_bicub.save(save_dir + "bicubic.png")
      y_net.save(save_dir + modelname + ".png")
      image.save(save_dir + imgs[i])
      resized_im.save(save_dir + "downsampled.png")


