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
  filename_rw = input("Enter model file: ")
  filename = common_parameters.relative_path + filename_rw;

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
  
  files = glob.glob(common_parameters.relative_path + "valid/*.png")
  toolbar_width = len(files)//2
  
  # setup toolbar
  sys.stdout.write("[%s]" % (" " * toolbar_width))
  sys.stdout.flush()
  sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
  with open(common_parameters.relative_path + 'validation_' + filename_rw.split(".",1)[0] + '.csv', 'w', newline='') as file:
    wcsv = csv.writer(file)
    wcsv.writerow(["Index", "File", "Model PSNR", "Lanczos PSNR", "Bilinear PSNR", "Bicubic PSNR"])
    PSNRs = []
    toTensor = transforms.Compose([transforms.ToTensor()])
    
    for i in range(len(files)):
      image = Image.open(files[i])
      image = image if image.mode == "RGB" else image.convert("RGB")
      im, padding, original_width, original_height = compat_pad(image, 5)
      real = toTensor(im).unsqueeze(0)
      sz = im.size
      inputs = toTensor(transforms.Resize((sz[1]//2,sz[0]//2), transforms.InterpolationMode.BILINEAR)(im)).unsqueeze(0)
    
    
      inputs = inputs.to(device)
      real = real.to(device)
      
      rl_crop = transforms.functional.crop(real.squeeze(), padding[1], padding[0], original_height, original_width)
      with torch.no_grad():
        y_crop = transforms.functional.crop(net(inputs).squeeze(), padding[1], padding[0], original_height, original_width)
        y_crop = torch.clamp(y_crop, 0, 1)
        model_psnr = psnr(rl_crop,y_crop).item()
        PSNRs.append(model_psnr)
      
      in_crop = transforms.functional.crop(inputs.squeeze(), padding[1]//2, padding[0]//2, original_height//2, original_width//2)
      rl_crop = transforms.functional.crop(real.squeeze(), padding[1], padding[0], original_height, original_width)
      im2 = transforms.ToPILImage()(in_crop)
      rz_size = (im2.size[1]*2, im2.size[0]*2)
      y_lanz = toTensor(transforms.Resize(rz_size, transforms.InterpolationMode.LANCZOS)(im2)).to(device)
      y_blin = toTensor(transforms.Resize(rz_size, transforms.InterpolationMode.BILINEAR)(im2)).to(device)
      y_bcub = toTensor(transforms.Resize(rz_size, transforms.InterpolationMode.BICUBIC)(im2)).to(device)
      lanz_psnr = psnr(rl_crop,y_lanz).item()
      blin_psnr = psnr(rl_crop,y_blin).item()
      bcub_psnr = psnr(rl_crop,y_bcub).item()
      
      wcsv.writerow([i, files[i], model_psnr, lanz_psnr, blin_psnr, bcub_psnr])
      if i % 2 == 1:
        sys.stdout.write("#")
        sys.stdout.flush()

  sys.stdout.write("]\n") # this ends the progress bar
  print(PSNRs)
  print(sum(PSNRs)/len(PSNRs))
  print("Max: ", max(PSNRs), " at index: ", PSNRs.index(max(PSNRs)))
  print("Min: ", min(PSNRs), " at index: ", PSNRs.index(min(PSNRs)))
