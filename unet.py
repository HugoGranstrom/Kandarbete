import torch.nn as nn
import torch.nn.functional as F
import os
import torch

class CnnBlock(nn.Module):
  def __init__(self, in_channels, out_channels, skip_final_activation=False):
    super().__init__()
    self.skip_final_activation = skip_final_activation
    self.activation = F.relu
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
    self.skip = nn.Conv2d(in_channels, out_channels, 1)
  
  def forward(self, x):
    input_x = x
    x = self.activation(self.conv1(x))
    x = self.conv2(x)
    x = x + self.skip(input_x)
    if not self.skip_final_activation:
      x = self.activation(x)
    return x

class StackedBlocks(nn.Module):
  def __init__(self, in_channels, out_channels, n_blocks):
    super().__init__()
    self.n_blocks = n_blocks
    if n_blocks == 0: # one Conv2d
      self.blocks = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1), nn.ReLU())
    elif n_blocks == 1: # two Conv2d with a skip connection
      self.blocks = CnnBlock(in_channels, out_channels)
    elif n_blocks > 1:
      self.blocks = nn.Sequential(CnnBlock(in_channels, out_channels), *[CnnBlock(out_channels, out_channels) for i in range(n_blocks-1)])
      self.skip = nn.Conv2d(in_channels, out_channels, 1)
    else:
      raise ValueError("n_blocks must be larger than 0, it was:", n_blocks)

  def forward(self, x):
    input_x = x
    x = self.blocks(x)
    return self.skip(input_x) + x if self.n_blocks > 1 else x

class Encoder(nn.Module):
  def __init__(self, nchannels, n_blocks, scale_power):
    super().__init__()
    self.nchannels = nchannels
    self.initScaler = nn.Upsample(scale_factor=2**scale_power, mode='bilinear', align_corners=True)
    self.scale_power = scale_power
    self.pool = nn.MaxPool2d(2)
    self.cnn = nn.Conv2d(nchannels[scale_power], nchannels[scale_power]-3, 1) # remove 3 channels to make room for the original 3-channel RGB image
    self.blocks = nn.ModuleList([StackedBlocks(nchannels[i], nchannels[i+1], n_blocks) for i in range(len(nchannels)-1)])

  def forward(self, x):
    input_x = x
    features = []
    x = self.initScaler(x)
    for block in self.blocks[:self.scale_power]:
      x = block(x)
      features.append(x)
      x = self.pool(x)
    x = self.cnn(x)
    x = torch.cat([input_x, x], dim=1)
    for block in self.blocks[self.scale_power:]:
      x = block(x)
      features.append(x)
      x = self.pool(x)
    return features

class UpscaleBlock(nn.Module): # A*A*C -> 2A*2A*C/2
  def __init__(self, in_channels, out_channels):
    super().__init__()
    #self.upscaleLayer = nn.PixelShuffle(2) # A*A*C -> 2A*2A*C/4
    self.upscaleLayer = nn.Upsample(scale_factor=2, mode='nearest')
    self.pad = nn.ReflectionPad2d(1)
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=0, stride=1)
    self.activation = F.relu

  def forward(self, x):
    x = self.pad(self.upscaleLayer(x))
    x = self.conv1(x)
    # activation function here?
    #x = self.activation(x)
    return x

class Decoder(nn.Module):
  def __init__(self, nchannels, scale_power):
    super().__init__()
    self.nchannels = nchannels
    self.upconvs = nn.ModuleList([UpscaleBlock(nchannels[i], nchannels[i]//2) for i in range(len(nchannels)-1)])
    self.blocks = nn.ModuleList([CnnBlock(nchannels[i], nchannels[i+1]) for i in range(len(nchannels)-1)])
    self.finalBlock = CnnBlock(nchannels[-1], 3, skip_final_activation=True)
    #self.finalUpscale = UpscaleBlock(nchannels[-1], nchannels[-1])
    #self.finalBlock = CnnBlock(nchannels[-1], 3, skip_final_activation=True)

  def forward(self, x, encoder_features):
    for i in range(len(self.nchannels)-1):
      x = self.upconvs[i](x)
      x = torch.cat([x, encoder_features[i]], dim=1)
      x = self.blocks[i](x)
    #x = self.finalUpscale(x)
    x = torch.sigmoid(self.finalBlock(x))
    return x

class UNet(nn.Module):
  # Important! The side lengths of the input image must be divisible depth times by 2. Add padding to nearest multiple when evaluating
  # Safe size: current_size + current_size % 2**(len(nchannels)-1) 
  # Pad to safe size, then crop to correct upscaled size afterwards
  def __init__(self, depth=5, init_channels=64, n_blocks=1, scale_power=1):
    super().__init__()
    #nchannels=[64,128,256,512]
    self.nchannels = [init_channels * 2**i for i in range(depth)]
    self.nchannels = [init_channels // 2**i for i in range(scale_power, 0, -1)] + self.nchannels
    print("nchannels:", [3] + self.nchannels)
    self.encoder = Encoder([3] + self.nchannels, n_blocks, scale_power)
    self.decoder = Decoder(self.nchannels[::-1], scale_power) # reverse

  def forward(self, x):
    encoder_features = self.encoder(x)
    out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
    return out

from torchsummary import summary

if __name__ == "__main__":
  x = torch.randn(4, 3, 192, 256)
  net = UNet(depth=5, scale_power=2, n_blocks=0)
  summary(net, (3, 192, 256), batch_size=5)
  #y = net(x)
  print(y.shape)