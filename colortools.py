import torch

xyz_from_rgb = torch.tensor([[0.412453, 0.357580, 0.180423],
                            [0.212671, 0.715160, 0.072169],
                            [0.019334, 0.119193, 0.950227]])

# R G B view

rgb_from_xyz = torch.inverse(xyz_from_rgb)

white_ref = torch.tensor((0.95047, 1., 1.08883)).view(1, 3, 1, 1)

def lab2xyz(lab): # funkar!
  L, a, b = lab[:,0,:,:],lab[:,1,:,:],lab[:,2,:,:]
  
  y = (L + 16.) / 116.
  x = (a / 500.) + y
  z = y - (b / 200.)
  
  if torch.any(z<0):
    print("Any!", torch.sum(z<0))
  z[z<0]=0

  out = torch.stack([x, y, z], dim=1)
  mask = out > 0.2068966
  out[mask] = torch.pow(out[mask], 3.)
  out[~mask] = (out[~mask] - 16.0 / 116.) / 7.787
  
  # rescale to the reference white (illuminant)
  out = out * white_ref
  
  return out
    
def xyz2rgb(xyz): # funkar!
  shape = torch.tensor(xyz.shape)[ [0,2,3] ]
  shape = list(shape)
  xyz_ = torch.stack([xyz[:,0,:,:].view(-1), xyz[:,1,:,:].view(-1), xyz[:,2,:,:].view(-1)], dim=1)
  
  mul = torch.matmul(xyz_, rgb_from_xyz.T)
  
  x,y,z = mul[:,0].view(shape), mul[:,1].view(shape), mul[:,2].view(shape)

  arr = torch.stack([x,y,z],dim=1)
  
  
  mask = arr > 0.0031308
  arr[mask] = 1.055 * torch.pow(arr[mask], 1 / 2.4) - 0.055
  arr[~mask] = arr[~mask] / 12.92
  torch.clip(arr, 0, 1, out=arr)
  return arr

def lab2rgb(lab):
  lab_scaling = torch.tensor([100, 128, 128]).float().view(1, 3, 1, 1)
  x = lab * lab_scaling
  return xyz2rgb(lab2xyz(x))


def rgb2xyz(rgb):
  arr = rgb
  mask = rgb > 0.04045
  arr[mask] = torch.pow((arr[mask] + 0.055) / 1.055, 2.4)
  arr[~mask] = arr[~mask] / 12.92
  
  shape = arr.shape[ [0,2,3] ]
  rgb_ = torch.stack([arr[:,0,:,:].view(-1),arr[:,1,:,:].view(-1),arr[:,2,:,:].view(-1)], dim=1)
  
  mul = torch.matmul(rgb_, xyz_from_rgb)
  
  x,y,z = mul[:,0].view(shape), mul[:,1].view(shape), mul[:,2].view(shape)
  return torch.stack([x,y,z],dim=1)

# Bx3xHxW <-> (B*H*W)x3

def xyz2lab(xyz):
  arr = xyz

  arr = arr.view(-1,3) / white_ref

  mask = arr > 0.008856
  arr[mask] = torch.pow(arr[mask],1/3)
  arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

  x, y, z = xyz[:,0,:,:],xyz[:,1,:,:],xyz[:,2,:,:]

  # Vector scaling
  L = (116. * y) - 16.
  a = 500.0 * (x - y)
  b = 200.0 * (y - z)

  return torch.stack([L,a,b])

def rgb2lab(rgb):
  return xyz2lab(rgb2xyz(rgb))
