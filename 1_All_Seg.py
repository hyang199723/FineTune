# Segment all masks using base SAM to get the performance of base SAm
# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import heapq
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
N = 275 # Total number of images
# %% Functions
os.chdir(wk_dir)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def show_mask(mask, score, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    iou = score[0]
    ax.set_title(f"{iou:.3f}")

    
def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    
# Calculate the Intersection over Union (IoU) score between two masks.
# Mask: n*m*3 matrix with the mask parts being 128.
def calculate_iou(mask1, mask2):
    mask1 = mask1.sum(axis=2)
    mask2 = mask2.sum(axis=2)

    mask1 = np.where(mask1 == 128, 1, 0)
    mask2 = np.where(mask2 == 128, 1, 0)
    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))
    iou = intersection / union
    return iou

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_single_mask(mask, ax):
    img = np.ones((mask['segmentation'].shape[0], mask['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    m = mask['segmentation']
    #color_mask = np.concatenate([np.random.random(3), [0.35]])
    color_mask = [0.06952425, 0.46226365, 0.44556518, 0.75]
    img[m] = color_mask
    ax.imshow(img)
    iou = mask['predicted_iou']
    ax.set_title(f"{iou:.3f}")

def show_top_mask(masks, ax):
    mask1 = sorted_mask[0]
    show_single_mask(mask1, ax[0, 0])

    mask2 = sorted_mask[1]
    show_single_mask(mask2, ax[0, 1])

    mask3 = sorted_mask[2]
    show_single_mask(mask3, ax[1, 0])

    mask4 = sorted_mask[3]
    show_single_mask(mask4, ax[1, 1])
#%% Import SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda:1" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model = sam)

# %% Get all segmentations
input_dir = "Data/CropSolar/"
output_dir = "Outputs/SolarTop4/"
number = 1
inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
image = cv2.imread(inputpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)
sorted_mask = sorted(masks, key=lambda d: d['predicted_iou'], reverse = True) # Sort masks

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(image)
ax[0, 1].imshow(image)
ax[1, 0].imshow(image)
ax[1, 1].imshow(image)
show_top_mask(sorted_mask, ax)
fig.suptitle(f"Solar {number} top 4 segs", fontsize=16)
plt.tight_layout()
fig.subplots_adjust(top=0.93)
plt.show()

# %%Output all image segmentation results
input_dir = "Data/CropSolar/"
output_dir = "Outputs/SolarTop4/"

for number in range(1, N + 1):
    print(number)
    inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
    image = cv2.imread(inputpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    sorted_mask = sorted(masks, key=lambda d: d['predicted_iou'], reverse = True) # Sort masks
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(image)
    ax[0, 1].imshow(image)
    ax[1, 0].imshow(image)
    ax[1, 1].imshow(image)
    show_top_mask(sorted_mask, ax)
    fig.suptitle(f"Solar {number} top 4 segs", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(wk_dir + output_dir + "Frame_" + f"{number:03}" + ".png")
    plt.close(fig)
# end

# %% Using bounding box
input_dir = "Data/CropSolar/"
#output_dir = "Outputs/BoundingBox/"
number = 1
inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
image = cv2.imread(inputpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

input_dir = "Data/CropSolar/"
#output_dir = "Outputs/SolarBoundingBox/"
input_box = np.array([10, 10, 170, 150])

predictor = SamPredictor(sam)
predictor.set_image(image)
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=True,
)

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[2], plt.gca())
show_box(input_box, plt.gca())
plt.axis('off')
plt.show()
#### When input is box, result is not good

# %% Initialize a grid of points
# Input image shape: 170 * 205 * 3
# Number of points: 20
#output_dir = "Outputs/BoundingBox/"
number = 1
input_dir = "Data/CropSolar/"
inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
image = cv2.imread(inputpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor = SamPredictor(sam)
predictor.set_image(image)
# Iput points should be B * N * 2
# Input labels should be B * N
raw_points = np.array([[i, j] for i in range(20, 180, 40) for j in range(20, 220, 40)])
B = len(raw_points)
n = 1
points = torch.tensor(raw_points.reshape(B, n, 2), device=predictor.device)
labels = np.array([1] * len(points))
labels = torch.tensor(labels.reshape(B, n), device=predictor.device)
trans_coords = predictor.transform.apply_coords_torch(points, image.shape[:2])
masks, quality, _ = predictor.predict_torch(
    point_coords = trans_coords,
    point_labels = labels,
    multimask_output=False,
)
masks = masks.reshape(masks.shape[0], masks.shape[-2], masks.shape[-1]).cpu().numpy()
quality = quality.cpu().numpy()
# Get top 4 masks
idx = heapq.nlargest(4, range(len(quality)), quality.take)
top_mask = masks[idx]
top_score = quality[idx]
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask, score in zip(top_mask, top_score):
    show_mask(mask, score, plt.gca(), random_color=False)
plt.axis('off')
plt.show()
# %% Itereate over all images
input_dir = "Data/CropSolar/"
output_dir = "Outputs/Solar_GridPoints/"
predictor = SamPredictor(sam)
raw_points = np.array([[i, j] for i in range(20, 180, 40) for j in range(20, 220, 40)])
x = [itm[0] for itm in raw_points]
y = [itm[1] for itm in raw_points]
B = len(raw_points)
n = 1
N = 275
points = torch.tensor(raw_points.reshape(B, n, 2), device=predictor.device)
labels = np.array([1] * len(points))
labels = torch.tensor(labels.reshape(B, n), device=predictor.device)
trans_coords = predictor.transform.apply_coords_torch(points, image.shape[:2])
for number in range(1, N+1):
    inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
    image = cv2.imread(inputpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    masks, quality, _ = predictor.predict_torch(
        point_coords = trans_coords,
        point_labels = labels,
        multimask_output=False,
    )
    masks = masks.reshape(masks.shape[0], masks.shape[-2], masks.shape[-1]).cpu().numpy()
    quality = quality.cpu().numpy()
    idx = heapq.nlargest(4, range(len(quality)), quality.take)
    m1 = masks[idx[0]]
    m2 = masks[idx[1]]
    m3 = masks[idx[2]]
    m4 = masks[idx[3]]
    s1 = quality[idx[0]]
    s2 = quality[idx[1]]
    s3 = quality[idx[2]]
    s4 = quality[idx[3]]
    fig, ax = plt.subplots(2, 2)
    for aaa in ax.flat:
        aaa.imshow(image)
        aaa.scatter(y, x, color = "red", marker = ".", s = 10)
    show_mask(m1, s1, ax[0, 0], random_color=False)
    show_mask(m2, s2, ax[0, 1], random_color=False)
    show_mask(m3, s3, ax[1, 0], random_color=False)
    show_mask(m4, s4, ax[1, 1], random_color=False)
    fig.suptitle(f"Solar {number} top 4 segs", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(wk_dir + output_dir + "Frame_" + f"{number:03}" + ".png")
    plt.close(fig)

################################################
################################################
################################################
################################################
################################################
################################################
################################################
################################################
    
# Old code below

# %% Load image 1 - 9
image_number = range(1,10)
IOU = []
for i in image_number:
    image_names = "Frame_00" + str(i) + ".png"
    filepath = wk_dir + "Data/SolarPanel/" + image_names
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Generate masks
    masks = mask_generator.generate(image)
    # Search solar panels
    min_area = 3700
    max_area = 5500
    index = []
    for idx, mask in enumerate(masks):
        if mask['area'] > min_area and mask['area'] < max_area:
            index.append(idx)
    # Get masks
    panel_mask = [masks[i] for i in index]
    seg = np.zeros((256, 256))
    for mask in panel_mask:
        cur_seg = mask['segmentation']
        temp = cur_seg * 1
        truth = np.where(temp == 1)
        min_y = np.min(truth[0])
        max_y = np.max(truth[0])
        min_x = np.min(truth[1])
        max_x = np.max(truth[1])
        temp[min_y:(max_y+1), min_x:(max_x+1)] = 1
        seg[min_y:(max_y+1), min_x:(max_x+1)] = 1
        IOU.append(np.sum(np.logical_and(temp, cur_seg)) /
                    np.sum(np.logical_or(temp, cur_seg)))
        #seg = seg + temp
    plt.imshow(seg)
    out_name = image_names = "Frame_00" + str(i) + ".txt"
    np.savetxt(wk_dir + "/Labels/SolarPanel/" + out_name, seg)

#mask = np.loadtxt(wk_dir + "/Labels/SolarPanel/" + out_name)


# %%Segment based on color
low = (95, 100, 120)
high = (110, 120, 140)
mask = cv2.inRange(image, low, high)
plt.imshow(mask)

#%% Point prompt segmentation
predictor = SamPredictor(sam)
image_names = "Frame_" + str(177) + ".png"
filepath = wk_dir + "Data/SolarPanel/" + image_names
image = cv2.imread(filepath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
predictor.set_image(image)
point = np.array([[80, 70]])
label = np.array([1])
masks, scores, logits = predictor.predict(
    point_coords=point,
    point_labels=label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(point, label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()
#%%
masks = masks.reshape(256, 256)
seg = np.zeros((256, 256))
temp = masks * 1
truth = np.where(temp == 1)
min_y = np.min(truth[0])
max_y = np.max(truth[0])
min_x = np.min(truth[1])
max_x = np.max(truth[1])
seg[min_y:(max_y+1), min_x:(max_x+1)] = 1

iou = np.sum(np.logical_and(temp, seg)) / np.sum(np.logical_or(temp, seg))
    



#%%
#SAM_assisted generating
# Area = 80000 - 1
# Total area of the image: 65536
# Solar panel area: 5474
# %% Look at generated masks
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 

#%%
# Solar panel dim: 49 - 83; 49 - 173
# Solar panel area: 4216
point1 = np.array([[79, 174]])
label = np.array([1])
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(point1, label, plt.gca())
plt.axis('off')
plt.show()
# %% Loop through masks to search area between 4000-4600
min_area = 4000
max_area = 4600
index = []
for idx, mask in enumerate(masks):
    if mask['area'] > min_area and mask['area'] < max_area:
        index.append(idx)

# %% Get masks
panel_mask = [masks[i] for i in index]
seg = np.zeros((256, 256, 3))
for mask in panel_mask:
    cur_seg = mask['segmentation']
    temp = cur_seg * 1
    truth = np.where(temp == 1)
    min_y = np.min(truth[0])
    max_y = np.max(truth[0])
    min_x = np.min(truth[1])
    max_x = np.max(truth[1])
    seg[min_y:(max_y+1), min_x:(max_x+1)] = 1
    #seg = seg + temp
cv2.imwrite(seg, wk_dir + "/Labels/SolarPanel")

#%%
temp = cur_seg * 1
truth = np.where(temp == 1)
min_y = np.min(truth[0])
max_y = np.max(truth[0])
min_x = np.min(truth[1])
max_x = np.max(truth[1])

temp[min_y:(max_y+1), min_x:(max_x+1)] = 1
# %%
np.savetxt('myarray.txt', temp)

# %%
