#%% Depth anything import from hugging face
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# Processor and model
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)

# %% test
# prepare image for the model
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
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
input_dir = "Data/CropSolar/"
output_dir = "Outputs/DepthAnything/"
number = 1
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
plt.imshow(depth, cmap = 'inferno')

# %% Generate for all images and save as png images
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
input_dir = "Data/CropSolar/"
output_dir = "Outputs/DepthAnything_raw/"
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
    fig, ax = plt.subplots()
    ax.imshow(depth, cmap = 'inferno')
    fig.suptitle(f"Solar {number} depth", fontsize=16)
    fig.savefig(wk_dir + output_dir + "Frame_" + f"{number:03}" + ".png")
    plt.close(fig)

# %% Generate Depth Info for all images and save as csv file
# Final decision: use raw depth information on cropped solar between 60 and 145
input_dir = "Data/CropSolar/"
output_dir = "Outputs/DepthInfo_raw/"
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
    np.savetxt(wk_dir + output_dir + "Frame_" + f"{number:03}" + ".csv", depth, delimiter=",")

# %% Depth anything on raw, uncropped images
input_dir = "Data/SolarPanel/"
output_dir = "Outputs/DepthAnything_raw/"
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
leftr = [50, 160, 53, 88] # Left solar position
rightr = [60, 170, 200, 227] # Right solar position
# 38
input_dir = "Data/SolarPanel/"
number = 38
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

plt.figure(0)
plt.imshow(image)
plt.title("Frame 38 raw image")

# Left: leftr[0]:leftr[1], leftr[2]:leftr[3]
# Right: rightr[0]:rightr[1], rightr[2]:rightr[3]
raw_depth = prediction.squeeze().cpu().numpy()
temp_raw = raw_depth[leftr[0]:leftr[1], leftr[2]:leftr[3]].flatten()
plt.figure(1)
plt.hist(temp_raw)
plt.title("Frame 38 left panel raw depth")

depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
depth = depth.astype(np.uint8)
plt.figure(2)
plt.imshow(depth, cmap = 'inferno')

temp = depth[leftr[0]:leftr[1], leftr[2]:leftr[3]].flatten()
plt.figure(3)
plt.hist(temp)
plt.title("Frame 38 left panel relative depth")


#%%
# Frame 115
number = 9
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

plt.figure(0)
plt.imshow(image)
plt.title("Frame 115 raw image")

raw_depth = prediction.squeeze().cpu().numpy()
# Left: leftr[0]:leftr[1], leftr[2]:leftr[3]
# Right: rightr[0]:rightr[1], rightr[2]:rightr[3]

# Left panel
temp_raw = raw_depth[leftr[0]:leftr[1], leftr[2]:leftr[3]].flatten()
plt.figure(1)
plt.hist(temp_raw)
plt.title("Frame 115 left panel raw depth")

# Right panel
temp_raw = raw_depth[rightr[0]:rightr[1], rightr[2]:rightr[3]].flatten()
plt.figure(2)
plt.hist(temp_raw)
plt.title("Frame 115 right panel raw depth")


depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
depth = depth.astype(np.uint8)


# Left panel
temp_raw = depth[leftr[0]:leftr[1], leftr[2]:leftr[3]].flatten()
plt.figure(3)
plt.hist(temp_raw)
plt.title("Frame 115 left panel relative depth")

# Right panel
temp_raw = depth[rightr[0]:rightr[1], rightr[2]:rightr[3]].flatten()
plt.figure(4)
plt.hist(temp_raw)
plt.title("Frame 115 right panel relative depth")


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

plt.figure(0)
plt.imshow(image)
plt.title("Frame 134 Image")
# Left: leftr[0]:leftr[1], leftr[2]:leftr[3]
# Right: rightr[0]:rightr[1], rightr[2]:rightr[3]
raw_depth = prediction.squeeze().cpu().numpy()
temp_raw = raw_depth[leftr[0]:leftr[1], leftr[2]:leftr[3]].flatten()
plt.figure(1)
plt.hist(temp_raw)
plt.title("Frame 134 left panel raw depth")

depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
depth = depth.astype(np.uint8)

temp = depth[leftr[0]:leftr[1], leftr[2]:leftr[3]].flatten()
plt.figure(2)
plt.hist(temp)
plt.title("Frame 134 left panel relative depth")
# %% Calibrate solar panel positions
import matplotlib.patches as patches
#leftr = [52, 173, 50, 83] # Left solar position of raw solar panel
leftr = [20, 140, 14, 50] # Left solar position of cropped solar panel
rightr = [30, 145, 160, 192] # Right solar position of cropped solar panel
#rightr = [60, 175, 195, 230] # Right solar position of raw solar panel

# The rectangle extends from xy[0] to xy[0] + width 
# in x-direction and from xy[1] to xy[1] + height in y-direction.
xy_left = (leftr[2], leftr[0])
rect_left = patches.Rectangle(xy_left, leftr[3] - leftr[2], leftr[1] - leftr[0]
                              , linewidth=0.5, edgecolor='r', facecolor='none')
xy_right = (rightr[2], rightr[0])
rect_right = patches.Rectangle(xy_right, rightr[3] - rightr[2], rightr[1] - rightr[0]
                              , linewidth=0.5, edgecolor='r', facecolor='none')

number = 9
input_dir = "Data/CropSolar/"
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

#plt.subplot(1, 3, 1)
ax1 = plt.subplot(1, 3, 1)
ax1.imshow(image)
ax1.add_patch(rect_left)
ax1.add_patch(rect_right)
plt.title("Frame 9")
# Left: leftr[0]:leftr[1], leftr[2]:leftr[3]
# Right: rightr[0]:rightr[1], rightr[2]:rightr[3]
plt.subplot(1, 3, 2)
plt.imshow(image[leftr[0]:leftr[1], leftr[2]:leftr[3], :])
plt.title("Frame 9 left")

plt.subplot(1, 3, 3)
plt.imshow(image[rightr[0]:rightr[1], rightr[2]:rightr[3], :])
plt.title("Frame 9 right")
# %% Use Frame 101 to calibrate middle solar panel positions
# midr = [52, 173, 100, 140] # Middle position of raw image
midr = [22, 143, 65, 103] # Middle position of cropped image
xy_mid = (midr[2], midr[0])
rect_mid = patches.Rectangle(xy_mid, midr[3] - midr[2], midr[1] - midr[0]
                              , linewidth=0.5, edgecolor='r', facecolor='none')
number = 101
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

ax1 = plt.subplot(1, 2, 1)
ax1.imshow(image)
ax1.add_patch(rect_mid)
plt.title("Frame 101")

plt.subplot(1, 2, 2)
plt.imshow(image[midr[0]:midr[1], midr[2]:midr[3], :])
plt.title("Frame 101 mid")

# %% Perform depth analysis on all 275 images
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
input_dir = "Data/SolarPanel/"
#output_dir = "Outputs/DepthAnything_raw/"
N = 275
left_raw = []
mid_raw = []
right_raw = []

left_rel = []
mid_rel = []
right_rel = []
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
    # Raw
    raw_depth = prediction.squeeze().cpu().numpy()
    left = raw_depth[leftr[0]:leftr[1], leftr[2]:leftr[3]].flatten()
    middle = raw_depth[midr[0]:midr[1], midr[2]:midr[3]].flatten()
    right = raw_depth[rightr[0]:rightr[1], rightr[2]:rightr[3]].flatten()
    left_raw = left_raw + left.tolist()
    mid_raw = mid_raw + middle.tolist()
    right_raw = right_raw + right.tolist()
    # Relative
    depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    left = depth[leftr[0]:leftr[1], leftr[2]:leftr[3]].flatten()
    middle = depth[midr[0]:midr[1], midr[2]:midr[3]].flatten()
    right = depth[rightr[0]:rightr[1], rightr[2]:rightr[3]].flatten()
    left_rel = left_rel + left.tolist()
    mid_rel = mid_rel + middle.tolist()
    right_rel = right_rel + right.tolist()

left_raw = np.array(left_raw)
mid_raw = np.array(mid_raw)
left_raw = np.array(left_raw)
left_rel = np.array(left_rel)
mid_rel = np.array(mid_rel)
right_rel = np.array(right_rel)

# Get stats and dist
plt.figure(0)
plt.hist(left_raw)
plt.title("Left region raw depth")
plt.figure(1)
plt.hist(mid_raw)
plt.title("Middle region raw depth")
plt.figure(2)
plt.hist(right_raw)
plt.title("Right region raw depth")
plt.figure(3)
plt.hist(left_rel)
plt.title("Left region relative depth")
plt.figure(4)
plt.hist(mid_rel)
plt.title("Middle region relative depth")
plt.figure(5)
plt.hist(right_rel)
plt.title("Right region relative depth")

# %% Perform depth analysis on all 275 cropped solar images
input_dir = "Data/CropSolar/"
#output_dir = "Outputs/DepthAnything_raw/"
N = 275
left_raw = []
mid_raw = []
right_raw = []

left_rel = []
mid_rel = []
right_rel = []
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
    # Raw
    raw_depth = prediction.squeeze().cpu().numpy()
    left = raw_depth[leftr[0]:leftr[1], leftr[2]:leftr[3]].flatten()
    middle = raw_depth[midr[0]:midr[1], midr[2]:midr[3]].flatten()
    right = raw_depth[rightr[0]:rightr[1], rightr[2]:rightr[3]].flatten()
    left_raw = left_raw + left.tolist()
    mid_raw = mid_raw + middle.tolist()
    right_raw = right_raw + right.tolist()
    # Relative
    depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    left = depth[leftr[0]:leftr[1], leftr[2]:leftr[3]].flatten()
    middle = depth[midr[0]:midr[1], midr[2]:midr[3]].flatten()
    right = depth[rightr[0]:rightr[1], rightr[2]:rightr[3]].flatten()
    left_rel = left_rel + left.tolist()
    mid_rel = mid_rel + middle.tolist()
    right_rel = right_rel + right.tolist()

left_raw = np.array(left_raw)
mid_raw = np.array(mid_raw)
left_raw = np.array(left_raw)
left_rel = np.array(left_rel)
mid_rel = np.array(mid_rel)
right_rel = np.array(right_rel)
# %% 
# Get stats and dist
plt.figure(0)
plt.hist(left_raw)
plt.title("Left region raw depth")
plt.figure(1)
plt.hist(mid_raw)
plt.title("Middle region raw depth")
plt.figure(2)
plt.hist(right_raw)
plt.title("Right region raw depth")
plt.figure(3)
plt.hist(left_rel)
plt.title("Left region relative depth")
plt.figure(4)
plt.hist(mid_rel)
plt.title("Middle region relative depth")
plt.figure(5)
plt.hist(right_rel)
plt.title("Right region relative depth")

# %% Final decision: use raw depth information on cropped solar between 60 and 145

