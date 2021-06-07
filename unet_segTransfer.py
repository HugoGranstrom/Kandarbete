import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

class ImageEmbedding(nn.Module):
    def __init__(self):
        super(ImageEmbedding, self).__init__()
        self.segModel = models.segmentation.deeplabv3_resnet50(pretrained=True).eval()
        self.segModel.classifier.__delitem__(4) # remove last layer which takes 256 channels to num_classes 21
        self.segModel.requires_grad = False
        for param in self.parameters():
            param.requires_grad = False
        
    
    def forward(self, x):
        return self.segModel(x)["out"]

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

class UNet(nn.Module):
  # Important! The side lengths of the input image must be divisible depth times by 2. Add padding to nearest multiple when evaluating
  # Safe size: current_size + current_size % 2**(len(nchannels)-1) 
  # Pad to safe size, then crop to correct upscaled size afterwards
  def __init__(self, depth=5, init_channels=64, embedding_size=50):
    super().__init__()
    self.embedding = ImageEmbedding()
    self.head = nn.Sequential(
      CnnBlock(256, 256),
      CnnBlock(256, 128),
      CnnBlock(128, 64),
      CnnBlock(64, 16)
    )
    self.finalBlock = CnnBlock(19, 3, skip_final_activation=True)

  def forward(self, x):
    with torch.no_grad():
      embeddings = self.embedding(x).detach()
    head = self.head(embeddings)
    head = torch.cat([x, head], dim=1)
    out = self.finalBlock(head)
    return out

from torchsummary import summary
import time

if __name__ == "__main__":
  x = torch.randn(2, 3, 256, 256)
  net = UNet(depth=5, embedding_size=50)
  input_size = 128
  #summary(net, (3, input_size, input_size), -1)
  y = net(x)
  print(y.shape)
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
  start = time.monotonic()
  for _ in range(int(50)):
    net.zero_grad()
    x = torch.zeros(1, 3, input_size, input_size)
    loss = (1 - net(x)).mean()
    loss.backward()
    optimizer.step()
  end = time.monotonic()
  print("Took: ", end - start, "seconds")
    