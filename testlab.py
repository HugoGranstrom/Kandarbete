import dataset
import colortools
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from PIL import Image

device = "cpu"

im_input = Image.open("imgs/0a26c96cf5900ea9.jpg")

im_np = np.asarray(im_input, dtype=np.float32) / 255
im_xyz = color.rgb2xyz(im_np)
im_xyz = torch.from_numpy(im_xyz).permute(2, 0, 1).unsqueeze(0)
print(im_xyz.shape)
im_rgb = colortools.xyz2rgb(im_xyz, device)

im_lab_t = dataset.ToLabTensor()(im_input)
im_rgb_t = dataset.TorchLab2RGB(im_lab_t, device)
im_xyz_our = colortools.rgb2xyz(im_rgb_t.unsqueeze(0))
print("XYZ diff:", torch.mean(torch.abs(im_xyz - im_xyz_our)))

im_rgb_colortools = colortools.lab2rgb(im_lab_t, device).squeeze()

print("Difference:", torch.mean(abs(im_rgb_t - im_rgb_colortools)))

fig = plt.figure()

fig.add_subplot(3, 2, 1)
plt.imshow(im_input)

fig.add_subplot(3, 2, 2)
#plt.imshow(transforms.ToPILImage()(im_rgb_t))
print(im_rgb_t.mean())
print(im_rgb_colortools.mean())
plt.imshow(dataset.TorchLab2RGBImg(im_lab_t, device))

fig.add_subplot(3, 2, 3)
plt.imshow(transforms.ToPILImage()(im_rgb_colortools))

fig.add_subplot(3, 2, 4)
plt.imshow(transforms.ToPILImage()(im_rgb.squeeze()))

plt.show()





