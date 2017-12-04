import h5py
from matplotlib import pyplot as plt
import numpy as np

# Index of image to display
index = 0

# Read file
h5_plain = h5py.File("../../res/processed/plain_5/train.h5")
h5_normalized = h5py.File("../../res/processed/normalized_5/train.h5")
h5_gray = h5py.File("../../res/processed/gray_5/train.h5")

# Load image
image_plain = h5_plain["train_dataset"][index]
image_normalized = h5_normalized["train_dataset"][index]
image_gray = h5_gray["train_dataset"][index]

# Print Label
print(h5_plain["train_labels"][index])

# Plot all four images
fig, ax = plt.subplots(1, 3)
ax[0].imshow(image_plain, interpolation="nearest")
ax[1].imshow(image_normalized, interpolation="nearest")
ax[2].imshow(np.squeeze(image_gray), cmap="gray", interpolation="nearest")
plt.show()
