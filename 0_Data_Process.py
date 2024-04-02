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




# %% Crop intruder image. A total of 365 intruder images
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
input_dir = "Data/Intruder/"
output_dir = "Data/CropIntruder/"
N = 365
for number in range(1, N + 1):
    inputpath = wk_dir + input_dir + "Frame_" + f"{number:06}" + ".jpg"
    image = Image.open(inputpath)

    # Crop the top section of the image
    # (left, upper, right, lower)
    bounding_box = (0, 40, 1920, 1080)
    cropped_image = image.crop(bounding_box)
    outname = wk_dir + output_dir + "Frame_" + f"{number:06}" + ".jpg"
    cropped_image.save(outname)

# %% A larger crop on intruder image
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
input_dir = "Data/Intruder/"
output_dir = "Data/CropIntruder_2/"
N = 365
for number in range(1, N + 1):
    inputpath = wk_dir + input_dir + "Frame_" + f"{number:06}" + ".jpg"
    image = Image.open(inputpath)

    # Crop the top section of the image
    # (left, upper, right, lower)
    bounding_box = (0, 40, 1600, 1080)
    cropped_image = image.crop(bounding_box)
    outname = wk_dir + output_dir + "Frame_" + f"{number:06}" + ".jpg"
    cropped_image.save(outname)

# %%
