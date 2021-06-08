import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from net import *
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


def predict(image, net, device):
  with torch.no_grad():
    im, padding, original_width, original_height = compat_pad(image, common_parameters.depth)
    y = net(transforms.ToTensor()(im).unsqueeze(0).to(device)).squeeze()
    y = torch.clamp(y, 0, 1)
    y = transforms.functional.crop(y, 2**common_parameters.scale_power*padding[1], 2**common_parameters.scale_power*padding[0], 2**common_parameters.scale_power*original_height, 2**common_parameters.scale_power*original_width)
    im = transforms.ToPILImage()(y)
    return im

if __name__ == '__main__':


  filename = input("Enter model file: ");
  imf = input("Enter image file: ")
  if imf == "":
    raise ValueError("No image was given!")
    

  device_name = "none"
  if torch.cuda.is_available():
    while device_name != 'cuda' and device_name != 'cpu':
      device_name = input("Enter device ('cuda', 'cpu'):")
      if device_name == "":
        device_name = 'cuda'
  else:
    device_name = "cpu"

  device = torch.device(device_name)
  
  net = UNet(depth=common_parameters.depth, scale_power=common_parameters.scale_power, nblocks=common_parameters.nblocks)
  loadNetEval(filename, net, device)
  net.to(device)
  net.eval()
  
  x = Image.open(imf).convert("RGB")
  plt.imshow(x)
  plt.show(block=False)
  plt.pause(0.05)

  im = predict(x, net, device)
    
  im.save("result.png")
  plt.figure()
  plt.imshow(im)
  plt.show(block=False)
  plt.pause(0.05)
  
  fig = plt.figure()
  y = transforms.Resize((x.size[1]*(2**common_parameters.scale_power), x.size[0]*(2**common_parameters.scale_power)), transforms.InterpolationMode.LANCZOS)(x)
  plt.imshow(y)
  plt.show()
  y.save("lanczos.png")