import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# Read .mat file
mat = sio.loadmat("../../res/cropped/train_32x32.mat")

# Number of image to display
index = 50

# Take the image part from the numpy array
image = mat['X'][:, :, :, index]

# Convert image to gray scale
image_gray = rgb2gray(image)

# Normalize original image
pix = (255 - image) * 1.0 / 255.0
norm_image = pix - np.mean(pix, axis=0)

# Convert to gray scale the normalized image
norm_image_gray = rgb2gray(norm_image)

# Extract and print image label
label = mat['y'][index]
print(label)

# Plot all four images
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(image, interpolation="nearest")
ax[0, 1].imshow(image_gray, cmap="gray", interpolation="nearest")
ax[1, 0].imshow(norm_image, interpolation="nearest")
ax[1, 1].imshow(norm_image_gray, cmap="gray", interpolation="nearest")
plt.show()
