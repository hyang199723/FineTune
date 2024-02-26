# Multiple points prompt to SAM

#%% Library and import
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor, SamImageProcessor
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry

wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Import model from local
# model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# Import model from hugging face
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

#%% Help functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    
# %% One simple exmaple
image = cv2.imread(wk_dir + '/Images/truck.jpg')
raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Masks size should be (color_dim  * width * height)
masks_1 = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
# For plot showing
seg_1 = masks_1[0][0]
input_point_1 = np.array(input_points[0])
input_label_1 = np.array([1])
scores_1 = np.array(scores.cpu()[0][0])

for i, (mask, score) in enumerate(zip(seg_1, scores_1)):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    show_mask(mask, plt.gca())
    show_points(input_point_1, input_label_1, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  

# %% Input with multiple points
# The points can be obtained by passing a list of list of list to the processor 
# that will create corresponding torch tensors of dimension 4
input_points = [[[350, 400]]]  # 2D location of a window in the image
inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Masks size should be (color_dim  * width * height)
masks_2 = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
# For plot showing
seg_2 = masks_2[0][0]
input_point_2 = np.array(input_points[0])
input_label_2 = np.array([1])
scores_2 = np.array(scores.cpu()[0][0])

for i, (mask, score) in enumerate(zip(seg_2, scores_2)):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    show_mask(mask, plt.gca())
    show_points(input_point_2, input_label_2, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  
#%% Combine segmentation from above
    
full_seg = torch.cat((seg_1, seg_2))
full_scores = np.concatenate((scores_1, scores_2))
full_point = np.concatenate((input_point_1, input_point_2))
input_label = np.array([1, 1])
for i, (mask, score) in enumerate(zip(full_seg, full_scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    show_mask(mask, plt.gca())
    show_points(full_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  

#%% More points
iters = 100
x_points = np.random.uniform(0, 1800, iters).reshape(-1, 1)
y_points = np.random.uniform(0, 1200, iters).reshape(-1, 1)
points = np.concatenate((x_points, y_points), axis = 1)
full_seg = torch.empty(1, raw_image.shape[0], raw_image.shape[1])
full_scores = np.array([0])
input_label = np.array([1]*iters)
for i, point in enumerate(points):
    input = [[list(points[i,:])]]
    inputs = processor(raw_image, input_points=input, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Masks size should be (color_dim  * width * height)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores
    # For plot showing
    seg = masks[0][0]
    scores= np.array(scores.cpu()[0][0])

    full_seg = torch.cat((full_seg, seg))
    full_scores = np.concatenate((full_scores,scores ))
# Remove first seg and score
final_seg = full_seg[1:]
final_scores = full_scores[1:]
for i, (mask, score) in enumerate(zip(full_seg, full_scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    show_mask(mask, plt.gca())
    show_points(points, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  

# Select based on scores
threshold = 1

# %% Sam Image Processor
SIP = SamImageProcessor()

# %%
msk = masks[0]
SIP.filter_masks(seg, scores, raw_image.shape[0:2], raw_image)
# %%
