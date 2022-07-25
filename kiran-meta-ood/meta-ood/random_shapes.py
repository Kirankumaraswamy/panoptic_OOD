import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import random
import io
import cv2
import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
import time

s = time.time()
n = random.randint(2, 9)        # Number of possibly sharp edges
r = random.random() # magnitude of the perturbation from the unit circle,
# should be between 0 and 1
N = n*3+1 # number of points in the Path
# There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve

angles = np.linspace(0,2*np.pi,N)
codes = np.full(N,Path.CURVE4)
codes[0] = Path.MOVETO
verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r*np.random.random(N)+1-r)[:,None]

verts[-1,:] = verts[0,:] # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
path = Path(verts, codes)


fig = plt.figure(1)
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, facecolor='none', lw=2)
ax.add_patch(patch)

ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
ax.axis('off') # removes the axis to leave only the shape

with io.BytesIO() as buff:
    fig.savefig(buff, format='raw')

    buff.seek(0)
    data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
w, h = fig.canvas.get_width_height()
im = data.reshape((int(h), int(w), -1))[:,:,0]
plt.figure(1).clear()


th, im_th = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY_INV)

# Copy the thresholded image.
im_floodfill = im_th.copy()
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255)
# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv

random_width = random.randint(30, 300)
random_height = random.randint(30, 300)

im_out = torch.unsqueeze(torch.tensor(im_out), dim=0)
T = Resize(size=(random_height, random_width), interpolation=InterpolationMode.NEAREST)
im_out = T(im_out)
im_out = im_out.squeeze().numpy()

h, w = im_out.shape
mask = np.zeros((512, 1024), dtype="uint8")

end_width = 1024 - w
end_height = 512 - h
start_width = random.randint(0, end_width)
start_height = random.randint(0, end_height)

mask[start_height:start_height+h, start_width:start_width+w]  += im_out
mask[np.where(mask == 0)] = 1
mask[np.where(mask == 255)] = 0
e = time.time()
print(e -s)

plt.imshow(mask)
plt.show()