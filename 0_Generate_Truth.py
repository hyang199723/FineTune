# Generate mask from .txt file
# %% Packages
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio

#%% Read .txt file and convert to 3D mask
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
#wk_dir = "/Users/hongjianyang/SAS/FineTune/"
input_dir = "Labels/SolarPanel/"
output_dir = "Labels/SolarPanelPNG/"
# Load reference images
filename = wk_dir  + "Data/SolarPanel/Frame_001.png"
image = cv2.imread(filename)
ref_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#ref_image
# Load reference mask
filename = wk_dir + "Labels/SolarPanel/Frame_001.txt"
ref_mask = np.loadtxt(filename,delimiter=",", dtype=str)
mask = []
for i in range(len(ref_mask)):
    mask = mask + ref_mask[i].split()
mask = np.array([float(i) for i in mask])
temp = np.array(mask.reshape((ref_image.shape[0], ref_image.shape[1])))
ref_mask = np.zeros((ref_image.shape[0], ref_image.shape[1], 3))
ref_mask[:,:,0] = ref_mask[:,:,1] = ref_mask[:,:,2] = temp
# Split it into two smaller panels
out_mask = (ref_mask * 255).astype(np.uint8)
imageio.imwrite(wk_dir + output_dir + "Frame_001.png", out_mask)
# %% Read image
input_dir = "Data/SolarPanel/"
out_dir = "Labels/SolarPanelPNG/"
fn = wk_dir  + input_dir + "Frame_001.png"
image = cv2.imread(fn)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
# [50, 50], [80, 175]
# [165, 50], [177, 200]
plt.scatter(200, 177, color='green', marker='*', 
           s=50, edgecolor='white', linewidth=1.25)
# Create a mask
ref_mask_1 = np.zeros((image.shape[0], image.shape[1], 3))
ref_mask_2 = np.zeros((image.shape[0], image.shape[1], 3))
ref_mask_1[50:175, 50:80, :] = 1
ref_mask_2[50:200, 165:200, :] = 1
out_mask_1 = (ref_mask_1 * 255).astype(np.uint8)
out_mask_2 = (ref_mask_2 * 255).astype(np.uint8)
imageio.imwrite(wk_dir + output_dir + "Frame_001_1.png", out_mask_1)
imageio.imwrite(wk_dir + output_dir + "Frame_001_2.png", out_mask_2)
# %%
