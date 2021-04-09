import torch.nn as nn
import torch.nn.functional as F
import os
import torch

class CnnBlock(nn.Module):
  def __init__(self, in_channels, out_channels, batch_norm, skip_final_activation=False):
    super().__init__()
    self.skip_final_activation = skip_final_activation
    self.activation = F.relu
    self.batch_norm = batch_norm
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
  
  def forward(self, x):
    x = self.activation(self.conv1(x))
    if self.batch_norm:
      x = self.bn1(x)
    if not self.skip_final_activation:
      x = self.activation(self.conv2(x))
      if self.batch_norm:
        x = self.bn2(x)
    else:
      x = self.conv2(x) # when used as the last layer of the network we want to use another activation function
    return x

class Encoder(nn.Module):
  def __init__(self, nchannels, batch_norm):
    super().__init__()
    self.nchannels = nchannels
    self.pool = nn.MaxPool2d(2)
    self.blocks = nn.ModuleList([CnnBlock(nchannels[i], nchannels[i+1], batch_norm) for i in range(len(nchannels)-1)])

  def forward(self, x):
    features = []
    for block in self.blocks:
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
    self.bn = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    x = self.pad(self.upscaleLayer(x))
    x = self.conv1(x)
    # activation function here?
    #x = self.activation(x)
    return x

class Decoder(nn.Module):
  def __init__(self, nchannels, batch_norm):
    super().__init__()
    self.nchannels = nchannels
    self.upconvs = nn.ModuleList([UpscaleBlock(nchannels[i], nchannels[i]//2) for i in range(len(nchannels)-1)])
    self.blocks = nn.ModuleList([CnnBlock(nchannels[i], nchannels[i+1], batch_norm) for i in range(len(nchannels)-1)])
    self.finalUpscale = UpscaleBlock(nchannels[-1], nchannels[-1])
    self.finalBlock = CnnBlock(nchannels[-1], 3, skip_final_activation=True, batch_norm=False)

  def forward(self, x, encoder_features):
    for i in range(len(self.nchannels)-1):
      x = self.upconvs[i](x)
      x = torch.cat([x, encoder_features[i]], dim=1)
      x = self.blocks[i](x)
    x = self.finalUpscale(x)
    x = torch.sigmoid(self.finalBlock(x))
    return x

class UNet(nn.Module):
  # Important! The side lengths of the input image must be divisible depth times by 2. Add padding to nearest multiple when evaluating
  # Safe size: current_size + current_size % 2**(len(nchannels)-1) 
  # Pad to safe size, then crop to correct upscaled size afterwards
  def __init__(self, depth=4, init_channels=64, batch_norm=False):
    super().__init__()
    #nchannels=[64,128,256,512]
    self.nchannels = [init_channels * 2**i for i in range(depth)]
    self.encoder = Encoder([3] + self.nchannels, batch_norm)
    self.decoder = Decoder(self.nchannels[::-1], batch_norm) # reverse

  def forward(self, x):
    encoder_features = self.encoder(x)
    out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
    return out