import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class AdverserialModel(nn.Module):
  def __init__(this, high_res):
    super().__init__()
    this.model = nn.Sequential(
      nn.Conv2d(3, 16, 3,padding=1), # 3*3*3*16 = 432
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(16, 32, 3,padding=1, stride=2), # 4608
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(32, 64, 3,padding=1, stride=2), # 18 432
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(64, 128, 3,padding=1, stride=2), # 73 728
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(128, 256, 3,padding=1, stride=2), # 
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(256, 512, 3,padding=1, stride=2), # 
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(512, 1024, 3,padding=1, stride=2), #
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(1024, 2048, 3,padding=1, stride=2), #
      nn.LeakyReLU(0.2, inplace=True),
      
      
      nn.Flatten(),
      
      nn.Linear(int(2048*high_res*high_res/(4**7)), 1024), # 8 388 608
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(1024, 128),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(128, 1)
    )

  def forward(this, x):
    return this.model(x)


class VGG(nn.Module):
    """VGG/Perceptual Loss
    
    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output
    """
    def __init__(self, conv_index: str = '22'):

        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)
        #self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False
        for param in self.parameters():
          param.requires_grad = False


    def calcLoss(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute VGG/Perceptual loss between Super-Resolved and High-Resolution
        Parameters
        ----------
        sr : torch.Tensor
            Super-Resolved model output tensor
        hr : torch.Tensor
            High-Resolution image tensor
        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr
        """
        def _forward(x):
            #x = self.sub_mean(x)
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)

        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss

def perceptual_loss(real, fake, vgg):
  """Normalizes y and y_hat, runs them through vgg and compares intermediate layers and returns the perceptual loss"""
  mean = torch.tensor([0.485, 0.456, 0.406])
  std = torch.tensor([0.229, 0.224, 0.225]) # the biggest value that can be normalized to is 2.64
  normalize = transforms.Normalize(mean.tolist(), std.tolist())
  unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

  loss = vgg.calcLoss(normalize(fake), normalize(real))
  return loss


def sobel_filter(y, device):
  kernel_x = torch.tensor([[1, 0, -1],[2,0,-2],[1,0,-1]]).view(1,1,3,3).expand(3,-1,-1,-1).float().to(device)
  kernel_y = torch.tensor([[1, 2, 1],[0,0,0],[-1,-2,-1]]).view(1,1,3,3).expand(3,-1,-1,-1).float().to(device)
  Gx = F.conv2d(y, kernel_x, groups=y.shape[1])
  Gy = F.conv2d(y, kernel_y, groups=y.shape[1])
  return (Gx**2 + Gy**2 + 1e-8).sqrt()

def psnr(real, fake):
  return -10*torch.log10(F.mse_loss(real, fake))