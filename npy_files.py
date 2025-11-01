import numpy as np
import matplotlib.pyplot as plt
import os

# Path to one of your Cellpose segmentation files
file_path = "/home/ibab/image_info_project/cellpose_results/jw-Kontrolle1_(c1+c5)_seg.npy"

# Load the segmentation mask
mask = np.load(file_path, allow_pickle=True).item()  # Cellpose saves dicts inside .npy

# Check the keys (contents)
print(mask.keys())

# Usually you'll find keys like:
# 'masks', 'outlines', 'flows', 'img', etc.
# To visualize:
plt.figure(figsize=(6, 6))
plt.imshow(mask['masks'], cmap='nipy_spectral')
plt.title("Segmentation Mask")
plt.axis('off')
plt.show()