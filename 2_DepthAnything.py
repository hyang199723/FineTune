#%% Depth anything import from hugging face
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# %% test
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
depth = prediction.squeeze().cpu().numpy()
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.astype(np.uint8)
plt.imshow(depth, cmap = 'inferno')


# %% Try some solar panel image
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
input_dir = "Data/CropSolar/"
output_dir = "Outputs/DepthAnything/"
number = 1
inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
image = cv2.imread(inputpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(device)
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
plt.imshow(depth, cmap = 'inferno')

# %% Generate for all images
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
input_dir = "Data/CropSolar/"
output_dir = "Outputs/DepthAnything/"
image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)
N = 275
for number in range(1, N + 1):
    print(number)
    inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
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
    fig.suptitle(f"Solar {number} depth", fontsize=16)
    fig.savefig(wk_dir + output_dir + "Frame_" + f"{number:03}" + ".png")
    plt.close(fig)

# %% Depth anything on raw, uncropped images
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
input_dir = "Data/SolarPanel/"
output_dir = "Outputs/DepthAnything_raw/"
image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)
N = 275
for number in range(1, N + 1):
    print(number)
    inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
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
    fig.suptitle(f"Solar {number} depth", fontsize=16)
    fig.savefig(wk_dir + output_dir + "Frame_" + f"{number:03}" + ".png")
    plt.close(fig)
# %% Make depth plots for a few images

# 38
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
input_dir = "Data/SolarPanel/"
number = 38
inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
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

raw_depth = prediction.squeeze().cpu().numpy()
temp_raw = raw_depth[55:160, 56:96].flatten()
plt.hist(temp_raw)
plt.title("Frame 38 left panel raw depth")

depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
depth = depth.astype(np.uint8)
plt.imshow(depth, cmap = 'inferno')

temp = depth[55:160, 56:96].flatten()
plt.hist(temp)
plt.title("Frame 38 left panel relative depth")

# Plot whole image depth
plt.hist(raw_depth.flatten())
plt.title("Frame 38 whole image raw depth")

plt.hist(depth.flatten())
plt.title("Frame 38 whole image relative depth")

#%%
# Frame 115
number = 115
inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
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

raw_depth = prediction.squeeze().cpu().numpy()
temp_raw = raw_depth[60:170, 190:220].flatten()
plt.figure(0)
plt.hist(temp_raw)
plt.title("Frame 115 right panel raw depth")

depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
depth = depth.astype(np.uint8)

temp = depth[60:170, 190:220].flatten()
plt.figure(1)
plt.hist(temp)
plt.title("Frame 115 right panel relative depth")


# Plot whole image depth
plt.figure(2)
plt.hist(raw_depth.flatten())
plt.title("Frame 115 whole image raw depth")

plt.figure(3)
plt.hist(depth.flatten())
plt.title("Frame 115 whole image relative depth")

#%% 
# Frame 134
number = 134
inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
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

raw_depth = prediction.squeeze().cpu().numpy()
temp_raw = raw_depth[55:160, 56:96].flatten()
plt.figure(0)
plt.hist(temp_raw)
plt.title("Frame 134 left panel raw depth")

depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
depth = depth.astype(np.uint8)

temp = depth[55:160, 56:96].flatten()
plt.figure(1)
plt.hist(temp)
plt.title("Frame 134 left panel relative depth")


# Plot whole image depth
plt.figure(2)
plt.hist(raw_depth.flatten())
plt.title("Frame 134 whole image raw depth")

plt.figure(3)
plt.hist(depth.flatten())
plt.title("Frame 134 whole image relative depth")
# %% Frame 101
# Frame 101
number = 246
inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
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

raw_depth = prediction.squeeze().cpu().numpy()
temp_raw = raw_depth[55:160, 100:130].flatten()
plt.figure(0)
plt.hist(temp_raw)
plt.title("Frame 101 middle panel raw depth")

depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
depth = depth.astype(np.uint8)

temp = depth[55:160, 100:130].flatten()
plt.figure(1)
plt.hist(temp)
plt.title("Frame 101 middle panel relative depth")


# Plot whole image depth
plt.figure(2)
plt.hist(raw_depth.flatten())
plt.title("Frame 101 whole image raw depth")

plt.figure(3)
plt.hist(depth.flatten())
plt.title("Frame 101 whole image relative depth")
