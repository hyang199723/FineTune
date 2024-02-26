# Process dataset

#%% Libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
from PIL import Image
# %% Crop SolarPanel
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
input_dir = "Data/SolarPanel/"

for input_number in range(1,276):
    filepath = wk_dir + input_dir + "Frame_" + f"{input_number:03}" + ".png"

    img = Image.open(filepath)

    # Define the bounding box (left, upper, right, lower)
    bounding_box = (35, 30, 240, 200)

    # Crop the image
    cropped_image = img.crop(bounding_box)

    #plt.imshow(cropped_image)

    output_dir = "Data/CropSolar/"
    outname = wk_dir + output_dir + "Frame_" + f"{input_number:03}" + ".png"
    cropped_image.save(outname)




# %%
