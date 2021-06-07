import torch.nn as nn
import torch.nn.functional as F
import os
import torch
# 170 vs 250 sec vs 120s
class CnnBlock(nn.Module):
  def __init__(self, in_channels, out_channels, skip_final_activation=False):
    super().__init__()
    self.skip_final_activation = skip_final_activation
    self.activation = F.relu
    intermediate = max(3, in_channels//4)
    self.conv1 = nn.Conv2d(in_channels, intermediate, 1, stride=1, padding=0)
    self.conv2 = nn.Conv2d(intermediate, intermediate, 3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(intermediate, out_channels, 1, stride=1, padding=0)
    self.skip = nn.Conv2d(in_channels, out_channels, 1)
  
  def forward(self, x):
    input_x = x
    x = self.activation(self.conv1(x))
    x = self.activation(self.conv2(x))
    x = self.conv3(x)
    x = x + self.skip(input_x)
    if not self.skip_final_activation:
      x = self.activation(x)
    return x

class Encoder(nn.Module):
  def __init__(self, nchannels):
    super().__init__()
    self.nchannels = nchannels
    self.pool = nn.MaxPool2d(2)
    self.blocks = nn.ModuleList([CnnBlock(nchannels[i], nchannels[i+1]) for i in range(len(nchannels)-2)])
    self.blocks.append(CnnBlock(nchannels[-2], nchannels[-1]))

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
    self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    x = self.pad(self.upscaleLayer(x))
    x = self.conv1(x)
    x = self.activation(x)
    return x

class Decoder(nn.Module):
  def __init__(self, nchannels, embedding_size):
    super().__init__()
    self.nchannels = nchannels
    self.upconvs = nn.ModuleList([UpscaleBlock(nchannels[i], nchannels[i]//2) for i in range(len(nchannels)-1)])
    self.blocks = nn.ModuleList([CnnBlock(nchannels[i], nchannels[i+1]) for i in range(len(nchannels)-1)])
    self.finalBlock = CnnBlock(nchannels[-1], embedding_size)

  def forward(self, x, encoder_features):
    for i in range(len(self.nchannels)-1):
      x = self.upconvs[i](x)
      x = torch.cat([x, encoder_features[i]], dim=1)
      x = self.blocks[i](x)
      temp = encoder_features[i]
      del temp
      encoder_features[i] = None
    x = self.finalBlock(x)
    return x

class UNet(nn.Module):
  # Important! The side lengths of the input image must be divisible depth times by 2. Add padding to nearest multiple when evaluating
  # Safe size: current_size + current_size % 2**(len(nchannels)-1) 
  # Pad to safe size, then crop to correct upscaled size afterwards
  def __init__(self, depth=5, init_channels=64, embedding_size=50):
    super().__init__()
    #nchannels=[64,128,256,512]
    self.nchannels = [init_channels * 2**i for i in range(depth)]
    self.encoder = Encoder([1] + self.nchannels)
    self.decoder = Decoder(self.nchannels[::-1], embedding_size=embedding_size) # reverse
    self.colorizer = nn.Sequential(*[CnnBlock(embedding_size, embedding_size) for i in range(3)], nn.Conv2d(embedding_size, 3, 3, padding=1))

  def forward(self, x):
    encoder_features = self.encoder(x)
    embeddings = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
    out = self.colorizer(embeddings)
    return out

from torchsummary import summary
import time

if __name__ == "__main__":
  x = torch.randn(2, 1, 256, 256)
  net = UNet(depth=5, embedding_size=50)
  input_size = 128
  summary(net, (1, input_size, input_size), -1)
  y = net(x)
  print(y.shape)
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
  start = time.monotonic()
  for _ in range(int(50)):
    net.zero_grad()
    x = torch.zeros(1, 1, input_size, input_size)
    loss = (1 - net(x)).mean()
    loss.backward()
    optimizer.step()
  end = time.monotonic()
  print("Took: ", end - start, "seconds")