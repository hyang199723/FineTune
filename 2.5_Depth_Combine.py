# Filter the masks based on depth information
# Final decision: use raw depth information on cropped solar between 60 and 145
# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import pandas as pd
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
    ax.set_title(f"{score:.3f}")

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
# Mask: 2-D mask of either true or false
def calculate_iou(mask1, mask2):
    mask1 = np.where(mask1, 1, 0)
    mask2 = np.where(mask2, 1, 0)
    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2)) # denominator = min(mask1, mask2)
    iou = intersection / union
    return iou

# Calculate the Intersection over min(A,B) between two masks.
# Mask: 2-D mask of either true or false
def calculate_max_score(mask1, mask2):
    mask1 = np.where(mask1, 1, 0)
    mask2 = np.where(mask2, 1, 0)
    intersection = np.logical_and(mask1, mask2)
    inter1 = np.sum(np.logical_and(intersection, mask1)) / np.sum(mask1)
    inter2 = np.sum(np.logical_and(intersection, mask2)) / np.sum(mask2)
    score = max(inter1, inter2)
    return score

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

def show_top_mask(sorted_mask, ax):
    mask1 = sorted_mask[0]
    show_single_mask(mask1, ax[0, 0])

    mask2 = sorted_mask[1]
    show_single_mask(mask2, ax[0, 1])

    mask3 = sorted_mask[2]
    show_single_mask(mask3, ax[1, 0])

    mask4 = sorted_mask[3]
    show_single_mask(mask4, ax[1, 1])
    
    #mask5 = sorted_mask[4]
    #show_single_mask(mask5, ax[2, 0])

    #mask6 = sorted_mask[5]
    #show_single_mask(mask6, ax[2, 1])
#%% Import SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda:1" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model = sam)

# %% Read Depth information
# Depth should be between 60 and 145
input_dir = "Outputs/DepthInfo_raw/"
number = 1
inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".csv"
depth = np.array(pd.read_csv(inputpath, header=None))
truth = np.logical_and(depth >= 60, depth <= 145)
# Get all segmentations
input_dir = "Data/CropSolar/"
inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
image = cv2.imread(inputpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)
# First, combine with depth information and update total area
for i in range(len(masks)):
    masks[i]["segmentation"] = np.logical_and(masks[i]["segmentation"], truth)
    masks[i]['area'] = np.sum(masks[i]["segmentation"])
# Second, filter by area
masks = [d for d in masks if d['area'] > 400]
# Sort by area and filter by IoU score
masks = sorted(masks, key=lambda d: d['area'], reverse = True)
final_masks = []
threshold = 0.6
for i in range(len(masks)):
    cur = masks[i]
    add = True
    for j in range(i+1, len(masks)):
        compare = masks[j]
        if calculate_max_score(cur["segmentation"], compare["segmentation"]) > threshold:
            add = False
            break
    if add:
        final_masks.append(cur)
# Finally, sort by IoU score
sorted_mask = sorted(final_masks, key=lambda d: d['predicted_iou'], reverse = True) # Sort masks
    

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(image)
ax[0, 1].imshow(image)
ax[1, 0].imshow(image)
ax[1, 1].imshow(image)
show_top_mask(sorted_mask, ax)
fig.suptitle(f"Solar {number} top 4 segs", fontsize=8)
plt.tight_layout()
fig.subplots_adjust(top=0.93)
plt.show()

# %%Output all image segmentation results
output_dir = "Outputs/Solar_Depth_Combine/"
threshold = 0.6
N = 275
for number in range(1, N + 1):
    print(number)
    # Read depth information
    input_dir = "Outputs/DepthInfo_raw/"
    inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".csv"
    depth = np.array(pd.read_csv(inputpath, header=None))
    truth = np.logical_and(depth >= 60, depth <= 145)
    # Read image
    input_dir = "Data/CropSolar/"
    inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
    image = cv2.imread(inputpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    # First, combine with depth information and update total area
    for i in range(len(masks)):
        masks[i]["segmentation"] = np.logical_and(masks[i]["segmentation"], truth)
        masks[i]['area'] = np.sum(masks[i]["segmentation"])
    # Second, filter by area
    masks = [d for d in masks if d['area'] > 400]
    
    # Sort by area and filter by IoU score
    masks = sorted(masks, key=lambda d: d['area'], reverse = True)
    final_masks = []
    threshold = 0.6
    for i in range(len(masks)):
        cur = masks[i]
        add = True
        for j in range(i+1, len(masks)):
            compare = masks[j]
            if calculate_max_score(cur["segmentation"], compare["segmentation"]) > threshold:
                add = False
                break
        if add:
            final_masks.append(cur)
    # Finally, sort by IoU score
    sorted_mask = sorted(final_masks, key=lambda d: d['predicted_iou'], reverse = True) # Sort masks
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(image)
    ax[0, 1].imshow(image)
    ax[1, 0].imshow(image)
    ax[1, 1].imshow(image)
    show_top_mask(sorted_mask, ax)
    fig.suptitle(f"Solar {number} top 4 segs", fontsize=8)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(wk_dir + output_dir + "Frame_" + f"{number:03}" + ".png")
    plt.close(fig)
# end
# %%
