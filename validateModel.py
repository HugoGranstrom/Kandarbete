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

if __name__ == '__main__':
  filename = input("Enter model file: ");

  device_name = "none"
  if torch.cuda.is_available():
    while device_name != 'cuda' and device_name != 'cpu':
      device_name = input("Enter device ('cuda', 'cpu'):")
      if device_name == "":
        device_name = 'cuda'
        print('cuda')
  else:
    device_name = "cpu"

  device = torch.device(device_name)
  
  net = UNet(depth=5)
  loadNetEval(filename, net, device)
  net.to(device)
  net.eval()
  
  validationset = FolderSetFull("valid")
  files = validationset.files
  dataset = DataLoader(validationset, batch_size=1, num_workers = 4)
  
  
  toolbar_width = 40
  toolbar_skips = (int) (len(dataset)/toolbar_width)
  
  # setup toolbar
  sys.stdout.write("[%s]" % (" " * toolbar_width))
  sys.stdout.flush()
  sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
  with open('validation.csv', 'w', newline='') as file:
    wcsv = csv.writer(file)
    wcsv.writerow(["Index", "File", "Model PSNR", "Lanczos PSNR", "Bilinear PSNR"])
    PSNRs = []
    for i, data in enumerate(dataset):
      inputs, real = data
      inputs = inputs.to(device)
      real = real.to(device)
      
      with torch.no_grad():
        y = net(inputs)
        model_psnr = psnr(real,y).item()
        PSNRs.append(model_psnr)
      
      im = transforms.ToPILImage()(inputs.squeeze())
      to_tensor = transforms.ToTensor()
      y_lanz = to_tensor(transforms.Resize((im.size[1]*2, im.size[0]*2), transforms.InterpolationMode.LANCZOS)(im)).unsqueeze(0).to(device)
      y_blin = to_tensor(transforms.Resize((im.size[1]*2, im.size[0]*2), transforms.InterpolationMode.BILINEAR)(im)).unsqueeze(0).to(device)
      lanz_psnr = psnr(real,y_lanz).item()
      blin_psnr = psnr(real,y_blin).item()
      
      wcsv.writerow([i, files[i], model_psnr, lanz_psnr, blin_psnr])
      if i%toolbar_skips==0:
        sys.stdout.write("#")
        sys.stdout.flush()

  sys.stdout.write("]\n") # this ends the progress bar
  print(PSNRs)
  print(sum(PSNRs)/len(PSNRs))
  print("Max: ", max(PSNRs), " at index: ", PSNRs.index(max(PSNRs)))
  print("Min: ", min(PSNRs), " at index: ", PSNRs.index(min(PSNRs)))
