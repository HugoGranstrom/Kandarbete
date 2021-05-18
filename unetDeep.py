import torch.nn as nn
import torch.nn.functional as F
import os
import torch

class CnnBlock(nn.Module):
  def __init__(self, in_channels, out_channels, skip_final_activation=False, residual=True, kernel_size = 3, dual = True):
    super().__init__()
    self.skip_final_activation = skip_final_activation
    self.activation = F.relu
    if dual:
      self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size-1)//2)
      self.batch1 = nn.BatchNorm2d(in_channels)
    else:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2)
      self.batch1 = nn.BatchNorm2d(out_channels)
    if dual:
      self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2)
      self.batch2 = nn.BatchNorm2d(out_channels)
    self.dual = dual
    self.residual = residual
    
    if self.residual:
      self.skip = nn.Conv2d(in_channels, out_channels, 1)
  
  def forward(self, x):
    input_x = x
    x = self.activation(self.batch1(self.conv1(x)))
    if self.dual:
      x = self.batch2(self.conv2(x))
    if self.residual:
      x = x + self.skip(input_x)
    if not self.skip_final_activation:
      x = self.activation(x)
    return x

class CnnChunk(nn.Module): 
   def __init__(self, in_channels, out_channels, skip_final_activation=False, blocks=1):
    super().__init__()
    self.block_list = nn.Sequential(*[CnnBlock(in_channels, in_channels, skip_final_activation) for i in range(blocks-1)], CnnBlock(in_channels, out_channels, skip_final_activation))
    
   def forward(self, x):
    return self.block_list(x)

class Encoder(nn.Module):
  def __init__(self, nchannels, nblocks):
    super().__init__()
    self.nchannels = nchannels
    self.pool = nn.MaxPool2d(2)
    self.blocks = nn.ModuleList([CnnBlock(nchannels[0], nchannels[1], False, False, 7, False)] + [CnnChunk(nchannels[i], nchannels[i+1], False, nblocks[i-1]) for i in range(1,len(nchannels)-1)])

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

  def forward(self, x):
    x = self.pad(self.upscaleLayer(x))
    x = self.conv1(x)
    # activation function here?
    #x = self.activation(x)
    return x

class Decoder(nn.Module):
  def __init__(self, nchannels):
    super().__init__()
    self.nchannels = nchannels
    self.upconvs = nn.ModuleList([UpscaleBlock(nchannels[i], nchannels[i]//2) for i in range(len(nchannels)-1)])
    self.blocks = nn.ModuleList([CnnBlock(nchannels[i], nchannels[i+1], False, True, 3, True) for i in range(len(nchannels)-1)])
    self.finalUpscale = UpscaleBlock(nchannels[-1], nchannels[-1])
    self.finalBlock = CnnBlock(nchannels[-1], 3,  skip_final_activation=True, residual=False, kernel_size = 3)

  def forward(self, x, encoder_features):
    for i in range(len(self.nchannels)-1):
      x = self.upconvs[i](x)
      x = torch.cat([x, encoder_features[i]], dim=1)
      x = self.blocks[i](x)
      temp = encoder_features[i]
      encoder_features[i] = None
      del temp
    
    x = self.finalUpscale(x)
    x = torch.sigmoid(self.finalBlock(x))
    return x

class UNet(nn.Module):
  # Important! The side lengths of the input image must be divisible depth times by 2. Add padding to nearest multiple when evaluating
  # Safe size: current_size + current_size % 2**(len(nchannels)-1) 
  # Pad to safe size, then crop to correct upscaled size afterwards
  def __init__(self, depth=4, init_channels=64, nblocks=[3,4,5,2,1]):
    super().__init__()
    #nchannels=[64,128,256,512]
    self.nchannels = [init_channels * 2**i for i in range(depth)]
    self.encoder = Encoder([3] + self.nchannels, nblocks)
    self.decoder = Decoder(self.nchannels[::-1]) # reverse

  def forward(self, x):
    encoder_features = self.encoder(x)
    out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
    return out