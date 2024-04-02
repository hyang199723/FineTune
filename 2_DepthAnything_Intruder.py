#%% Depth anything import from hugging face
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# %% Try one intruder image
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
input_dir = "Data/"
output_dir = "Outputs/Intruder/"
number = 1
#inputpath = wk_dir + input_dir + "Frame_" + f"{number:06}" + ".jpg"
inputpath = wk_dir + input_dir + "Mirror.png"
image = cv2.imread(inputpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)

inputs = image_processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.shape[0:2],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
depth = prediction.squeeze().cpu().numpy()
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.astype(np.uint8)
plt.imshow(depth, cmap = 'inferno')
# %% Generate for all images
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
input_dir = "Data/CropIntruder_2/"
output_dir = "Outputs/Intruder_2/"
image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)
N = 365
for number in range(1, N + 1):
    print(number)
    inputpath = wk_dir + input_dir + "Frame_" + f"{number:06}" + ".jpg"
    image = cv2.imread(inputpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.shape[0:2],
        mode="bicubic",
        align_corners=False,
    )

    depth = prediction.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(depth, cmap = 'inferno')
    fig.suptitle(f"Intruder {number} depth", fontsize=16)
    fig.savefig(wk_dir + output_dir + "Frame_" + f"{number:06}" + ".jpg")
    plt.close(fig)

# %%
