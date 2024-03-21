# Combine the output masks as defined by IOU score
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
number = 1
inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
image = cv2.imread(inputpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)
# Calculate all IoU scores
final_masks = []
threshold = 0.9
for i in range(len(masks)):
    cur = masks[i]
    add = True
    for j in range(i+1, len(masks)):
        compare = masks[j]
        if calculate_iou(cur["segmentation"], compare["segmentation"]) > 0.3:
            add = False
            break
    if add:
        final_masks.append(cur)

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
output_dir = "Outputs/Solar_Combined_Top4/"
threshold = 0.3
for number in range(1, N + 1):
    print(number)
    inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
    image = cv2.imread(inputpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    final_masks = []
    threshold = 0.3
    for i in range(len(masks)):
        cur = masks[i]
        add = True
        for j in range(i+1, len(masks)):
            compare = masks[j]
            if calculate_iou(cur["segmentation"], compare["segmentation"]) > threshold:
                add = False
                break
        if add:
            final_masks.append(cur)
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
    multimask_output=True,
)
dup_points = np.repeat(raw_points, 3, axis = 0)
# Get a single mask
#mask = masks[0]
masks = masks.reshape(masks.shape[0]*masks.shape[1], masks.shape[-2], masks.shape[-1]).cpu().numpy()
quality = quality.reshape(quality.shape[0]*quality.shape[1]).cpu().numpy()
# calculate iou scores
final_masks_idx = []
threshold = 0.9
for i in range(len(masks)):
    cur = masks[i]
    add = True
    for j in range(i+1, len(masks)):
        compare = masks[j]
        if calculate_iou(cur, compare) > threshold:
            add = False
            break
    if add:
        final_masks_idx.append(i)
filtered_mask = masks[final_masks_idx]
filtered_score = quality[final_masks_idx]
filtered_points = dup_points[final_masks_idx]
# Get top 4 masks
idx = heapq.nlargest(4, range(len(filtered_score)), filtered_score.take)
top_mask = filtered_mask[idx]
top_score = filtered_score[idx]
top_coords = filtered_points[idx]
#top_coords = points[idx]
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask, score, coords in zip(top_mask, top_score, top_coords):
    show_mask(mask, score, plt.gca(), random_color=False)
    # def show_points(coords, labels, ax, marker_size=20):
    plt.scatter(coords[0], coords[1], color='green', marker='*', 
               s=200, edgecolor='white', linewidth=1.25)
plt.axis('off')
plt.show()

# %% Itereate over all images
input_dir = "Data/CropSolar/"
output_dir = "Outputs/Solar_Combined_GridPoints/"
predictor = SamPredictor(sam)
raw_points = np.array([[i, j] for i in range(20, 180, 40) for j in range(20, 220, 40)])
dup_points = np.repeat(raw_points, 3, axis = 0)
x = [itm[0] for itm in raw_points]
y = [itm[1] for itm in raw_points]
B = len(raw_points)
n = 1
N = 275
points = torch.tensor(raw_points.reshape(B, n, 2), device=predictor.device)
labels = np.array([1] * len(points))
labels = torch.tensor(labels.reshape(B, n), device=predictor.device)
trans_coords = predictor.transform.apply_coords_torch(points, image.shape[:2])
threshold = 0.9
for number in range(1, N+1):
    inputpath = wk_dir + input_dir + "Frame_" + f"{number:03}" + ".png"
    image = cv2.imread(inputpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    masks, quality, _ = predictor.predict_torch(
        point_coords = trans_coords,
        point_labels = labels,
        multimask_output=True,
    )
    masks = masks.reshape(masks.shape[0]*masks.shape[1], masks.shape[-2], masks.shape[-1]).cpu().numpy()
    quality = quality.reshape(quality.shape[0]*quality.shape[1]).cpu().numpy()
    # calculate iou scores
    final_masks_idx = []
    for i in range(len(masks)):
        cur = masks[i]
        add = True
        for j in range(i+1, len(masks)):
            compare = masks[j]
            if calculate_iou(cur, compare) > threshold:
                add = False
                break
        if add:
            final_masks_idx.append(i)
    filtered_mask = masks[final_masks_idx]
    filtered_score = quality[final_masks_idx]
    filtered_points = dup_points[final_masks_idx]
    idx = heapq.nlargest(4, range(len(filtered_score)), filtered_score.take)
    m1 = filtered_mask[idx[0]]
    m2 = filtered_mask[idx[1]]
    m3 = filtered_mask[idx[2]]
    m4 = filtered_mask[idx[3]]
    s1 = filtered_score[idx[0]]
    s2 = filtered_score[idx[1]]
    s3 = filtered_score[idx[2]]
    s4 = filtered_score[idx[3]]
    c1 = filtered_points[idx[0]]
    c2 = filtered_points[idx[1]]
    c3 = filtered_points[idx[2]]
    c4 = filtered_points[idx[3]]
    fig, ax = plt.subplots(2, 2)
    for aaa in ax.flat:
        aaa.imshow(image)
        aaa.scatter(y, x, color = "red", marker = ".", s = 10)
    show_mask(m1, s1, ax[0, 0], random_color=False)
    ax[0, 0].scatter(c1[0], c1[1], color='green', marker='*', 
               s=200, edgecolor='white', linewidth=1.25)
    show_mask(m2, s2, ax[0, 1], random_color=False)
    ax[0, 1].scatter(c2[0], c2[1], color='green', marker='*', 
               s=200, edgecolor='white', linewidth=1.25)
    show_mask(m3, s3, ax[1, 0], random_color=False)
    ax[1, 0].scatter(c3[0], c3[1], color='green', marker='*', 
               s=200, edgecolor='white', linewidth=1.25)
    show_mask(m4, s4, ax[1, 1], random_color=False)
    ax[1, 1].scatter(c4[0], c4[1], color='green', marker='*', 
               s=200, edgecolor='white', linewidth=1.25)
    fig.suptitle(f"Solar {number} top 4 segs", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(wk_dir + output_dir + "Frame_" + f"{number:03}" + ".png")
    plt.close(fig)
# %%
