import matplotlib.pyplot as plt
from dataset import *
from net import *
from unet import *

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


def predict(image, net, device):
  with torch.no_grad():
    im, padding, original_width, original_height = compat_pad(image, 4)
    y = net(transforms.ToTensor()(im).unsqueeze(0).to(device)).squeeze()
    y = transforms.functional.crop(y, 2*padding[1], 2*padding[0], 2*original_height, 2*original_width)
    im = transforms.ToPILImage()(y)
    return im

if __name__ == '__main__':
  imf = input("Enter file: ")
  if imf == "":
    OpenDataset([],1).download_image("0a2cc77c7437e2fb")
    imf = "imgs/0a2cc77c7437e2fb.jpg"
    

  if torch.cuda.is_available():
    device_name = "none"
    while device_name != 'cuda' and device_name != 'cpu':
      device_name = input("Enter device ('cuda', 'cpu'):")
      if device_name == "":
        device_name = 'cuda'
  else:
    device_name = "cpu"

  device = torch.device(device_name)

  filename = "net_UNet.pt"

  factor_s = input("Enter dimension upscale factor: 2^")
  if factor_s == "":
    factor = 1
    print("=2")
  else:
    factor = int(factor_s)
    print("=",2**factor)
  
  net = UNet(depth=5-int(factor_s), scale_power=int(factor_s), n_blocks=1, init_channels=64*2**int(factor_s))
  loadNetEval(filename, net, device)
  #loadNetEval("/content/drive/MyDrive/Colab Notebooks/" + filename, net, device)
  net.to(device)
  net.eval()
  
  x = Image.open(imf).convert("RGB")
  #x = Image.open("CAM00017.jpg").convert("RGB")
  plt.imshow(x)
  plt.show(block=False)
  plt.pause(0.05)
  
  #y = net(transforms.ToTensor()(x).unsqueeze(0).to(device))
  #im = transforms.ToPILImage()(y.squeeze())
  im = predict(x, net, device)
    
  im.save("result.png")
  plt.figure()
  plt.imshow(im)
  plt.show(block=False)
  plt.pause(0.05)
  plt.figure()

  y = transforms.Resize((x.size[1]*(2**factor), x.size[0]*(2**factor)), transforms.InterpolationMode.LANCZOS)(x)
  plt.imshow(y)
  plt.show()
  y.save("lanczos.png")