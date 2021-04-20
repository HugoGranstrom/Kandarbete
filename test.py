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
    


  print('cuda' if torch.cuda.is_available() else 'cpu')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  device = 'cpu'

  filename = "GAN_UNet_v1.pt"

  net = UNet(depth=5)
  loadNetEval(filename, net, device)
  #loadNetEval("/content/drive/MyDrive/Colab Notebooks/" + filename, net, device)
  net.to(device)
  net.eval()
  with torch.no_grad():
    x = Image.open(imf).convert("RGB")
    #x = Image.open("CAM00017.jpg").convert("RGB")
    plt.imshow(x)
    plt.show(block=False)
    plt.pause(0.05)
    plt.figure()
    #y = net(transforms.ToTensor()(x).unsqueeze(0).to(device))
    #im = transforms.ToPILImage()(y.squeeze())
    im = predict(x, net, device)
    im.save("result.png")
    plt.imshow(im)
    plt.show(block=False)
    plt.pause(0.05)
    plt.figure()

  y = transforms.Resize((x.size[1]*2, x.size[0]*2), transforms.InterpolationMode.LANCZOS)(x)
  plt.imshow(y)
  plt.show()
  y.save("lanczos.png")