import torch.nn as nn
import torch.nn.functional as F
import os
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 15, stride=1, padding=7, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv2d(64, 128, 5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv4 = nn.Conv2d(128, 256, 5, stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.pixelShuffle = nn.PixelShuffle(2)
        self.conv5 = nn.Conv2d(64, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv6 = nn.Conv2d(16, 3, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
				
        self.leaky = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        x = self.leaky(self.conv3(x))
        x = self.leaky(self.conv4(x))
        x = self.leaky(self.pixelShuffle(x))
        x = self.leaky(self.conv5(x))
        x = 4*torch.tanh(self.conv6(x))
        
        return x



def saveNet(filename, net, optimizer, iterations, train_loss, val_loss):
  torch.save({
      "net": net.state_dict(),
      "optimizer": optimizer.state_dict(),
      "iteration": iterations,
      "loss": train_loss,
      "val_loss": val_loss
  }, filename)

def loadNet(filename, net, optimizer, device):
  try:
    checkpoint = torch.load(filename, map_location=device)
    net.load_state_dict(checkpoint["net"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]
    train_loss = checkpoint["loss"]
    val_loss = checkpoint["val_loss"]
    print(f"Net loaded from memory! Starting on iteration {iteration[-1]+1} with train-loss {train_loss[-1]}")
    return iteration, train_loss, val_loss
  except (OSError, FileNotFoundError):
    print(f"Couldn't find {filename}, creating new net!")
    return [], [], []

def loadNetEval(filename, net, device):
  try:
    checkpoint = torch.load(filename, map_location=device)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    print(filename, "successfully loaded in eval mode")
  except (OSError, FileNotFoundError):
    print(f"Couldn't find {filename}")