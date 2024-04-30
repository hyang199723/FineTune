# Personalize SAM
#%% Packages
import torch
from PIL import Image

import requests
from transformers import SamModel, SamProcessor, SamImageProcessor, AutoProcessor

import numpy as np
import matplotlib.pyplot as plt
import cv2

from segment_anything import sam_model_registry
from huggingface_hub import hf_hub_download 
from PIL import Image 
from torch import nn 
from torch.optim import AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR 
import torch.nn.functional as F 
from torchvision.transforms.functional import resize, to_pil_image 
from typing import Tuple 
device = "cuda:1" if torch.cuda.is_available() else "cpu" 
# %% Training free version of PerSAM
processor = AutoProcessor.from_pretrained("facebook/sam-vit-huge") 
model = SamModel.from_pretrained("facebook/sam-vit-huge")
#wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/FineTune/"
wk_dir = "/Users/hongjianyang/SAS/FineTune/"
input_dir = "Labels/SolarPanel/"
# Load reference images
filename = hf_hub_download(repo_id="nielsr/persam-dog", filename="dog.jpg", repo_type="dataset")
ref_image = Image.open(filename).convert("RGB")
ref_image

filename = hf_hub_download(repo_id="nielsr/persam-dog", filename="dog_mask.png", repo_type="dataset")
ref_mask = cv2.imread(filename)
ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
gt_mask = gt_mask.float().unsqueeze(0).flatten(1)


visual_mask = ref_mask.astype(np.uint8)
Image.fromarray(visual_mask)

filename = hf_hub_download(repo_id="nielsr/persam-dog", filename="new_dog.jpg", repo_type="dataset")
test_image = Image.open(filename).convert("RGB").convert("RGB")
test_image

# %% Get target embedding
model.to(device) # Step 1: Image features encoding 
inputs = processor(images=ref_image, return_tensors="pt").to(device) 
pixel_values = inputs.pixel_values 
with torch.no_grad(): 
    # Forward run of reference image through SAM vision encoder
    image_embeddings = model.get_image_embeddings(pixel_values)
    ref_feat = image_embeddings.squeeze().permute(1, 2, 0)

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]: 
    """ Compute the output size given input size and target long side length. """ 
    scale = long_side_length * 1.0 / max(oldh, oldw) 
    newh, neww = oldh * scale, oldw * scale 
    neww = int(neww + 0.5) 
    newh = int(newh + 0.5) 
    return (newh, neww) 
def preprocess(x: torch.Tensor, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375], img_size=1024) -> torch.Tensor: 
    """Normalize pixel values and pad to a square input.""" 
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1) 
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1) # Normalize colors 
    x = (x - pixel_mean) / pixel_std # Pad 
    h, w = x.shape[-2:] 
    padh = img_size - h 
    padw = img_size - w 
    x = F.pad(x, (0, padw, 0, padh)) 
    return x 
def prepare_mask(image, target_length=1024): 
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length) 
    mask = np.array(resize(to_pil_image(image), target_size)) 
    input_mask = torch.as_tensor(mask) 
    input_mask = input_mask.permute(2, 0, 1).contiguous()[None, :, :, :] 
    input_mask = preprocess(input_mask) 
    return input_mask

# Step 2: interpolate reference mask 
# Reference mask is the target (0,1)
ref_mask = prepare_mask(ref_mask) 
ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear") 
ref_mask = ref_mask.squeeze()[0] 
# Step 3: Target feature extraction 
target_feat = ref_feat[ref_mask > 0] 
target_feat_mean = target_feat.mean(0) 
target_feat_max = torch.max(target_feat, dim=0)[0] 
target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

# %% Get cosine similartiy
h, w, C = ref_feat.shape 
target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True) 
ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True) 
ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w) 
sim = target_feat @ ref_feat

sim = sim.reshape(1, 1, h, w) 
sim = F.interpolate(sim, scale_factor=4, mode="bilinear") 
sim = processor.post_process_masks(sim.unsqueeze(1), 
                                   original_sizes=inputs["original_sizes"].tolist(), 
                                   reshaped_input_sizes=inputs["reshaped_input_sizes"].tolist(), 
                                   binarize=False) 
sim = sim[0].squeeze()

# %% Location Prior
def point_selection(mask_sim, topk=1): 
    # Top-1 point selection 
    w, h = mask_sim.shape 
    topk_xy = mask_sim.flatten(0).topk(topk)[1] 
    topk_x = (topk_xy // h).unsqueeze(0) 
    topk_y = (topk_xy - topk_x * h) 
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0) 
    topk_label = np.array([1] * topk) 
    topk_xy = topk_xy.cpu().numpy() 
    return topk_xy, topk_label 
# Positive location prior 
topk_xy, topk_label = point_selection(sim, topk=1) 
print("Topk_xy:", topk_xy) 
print("Topk_label:", topk_label)

# %% Optimize mask weights
def calculate_dice_loss(inputs, targets, num_masks = 1): 
    """ Compute the DICE loss, similar to generalized IOU for masks 
    Args: inputs: A float tensor of arbitrary shape. The predictions for each example. 
    targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class). """ 
    inputs = inputs.sigmoid() 
    inputs = inputs.flatten(1) 
    numerator = 2 * (inputs * targets).sum(-1) 
    denominator = inputs.sum(-1) + targets.sum(-1) 
    loss = 1 - (numerator + 1) / (denominator + 1) 
    return loss.sum() / num_masks 
def calculate_sigmoid_focal_loss(inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2): 
    """ Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002. 
    Args: inputs: A float tensor of arbitrary shape. The predictions for each example. 
    targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class). 
    alpha: (optional) Weighting factor in range (0,1) to balance positive vs negative examples. 
    Default = -1 (no weighting). gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. 
    Returns: Loss tensor """ 
    prob = inputs.sigmoid() 
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") 
    p_t = prob * targets + (1 - prob) * (1 - targets) 
    loss = ce_loss * ((1 - p_t) ** gamma) 
    if alpha >= 0: 
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets) 
        loss = alpha_t * loss 
        return loss.mean(1).sum() / num_masks 
def postprocess_masks(masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...], img_size=1024) -> torch.Tensor: 
    """ Remove padding and upscale masks to the original image size. 
    Arguments: masks (torch.Tensor): Batched masks from the mask_decoder, in BxCxHxW format. 
    input_size (tuple(int, int)): The size of the image input to the model, in (H, W) format. Used to remove padding. 
    original_size (tuple(int, int)): The original size of the image before resizing for input to the model, in (H, W) format. 
    Returns: (torch.Tensor): Batched masks in BxCxHxW format, where (H, W) is given by original_size. """ 
    masks = F.interpolate( masks, (img_size, img_size), mode="bilinear", align_corners=False, ) 
    masks = masks[..., : input_size[0], : input_size[1]] 
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False) 
    return masks

class Mask_Weights(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3) # Learnable mask weights 


mask_weights = Mask_Weights() 
mask_weights.to(device) 
mask_weights.train()         
num_epochs = 1000 
log_epoch = 200 
optimizer = AdamW(mask_weights.parameters(), lr=1e-3, eps=1e-4) 
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

inputs = processor(ref_image, input_points=[topk_xy.tolist()], 
                   input_labels=[topk_label.tolist()], return_tensors="pt").to(device) 
for k,v in inputs.items(): 
    print(k,v.shape)


gt_mask = gt_mask.to(device) 
for train_idx in range(num_epochs): # Run the decoder 
    with torch.no_grad(): 
        outputs = model( input_points=inputs.input_points, input_labels=inputs.input_labels, 
                        image_embeddings=image_embeddings.to(device), multimask_output=True, ) 
    logits_high = postprocess_masks(masks=outputs.pred_masks.squeeze(1), 
                                    input_size=inputs.reshaped_input_sizes[0].tolist(), original_size=inputs.original_sizes[0].tolist()) 
    logits_high = logits_high[0].flatten(1) # Weighted sum three-scale masks 
    weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0) 
    logits_high = logits_high * weights 
    logits_high = logits_high.sum(0).unsqueeze(0) 
    dice_loss = calculate_dice_loss(logits_high, gt_mask) 
    focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask) 
    loss = dice_loss + focal_loss 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
    scheduler.step() 
    if train_idx % log_epoch == 0: 
        print('Train Epoch: {:} / {:}'.format(train_idx, num_epochs)) 
        current_lr = scheduler.get_last_lr()[0] 
        print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))

mask_weights.eval() 
weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0) 
weights_np = weights.detach().cpu().numpy() 
print('======> Mask weights:\n', weights_np)

#%% Test Image
filename = hf_hub_download(repo_id="nielsr/persam-dog", filename="new_dog.jpg", repo_type="dataset") 
test_image = Image.open(filename).convert("RGB").convert("RGB") 
test_image

# prepare for the model 
inputs = processor(images=test_image, return_tensors="pt").to(device) # Image feature encoding 
with torch.no_grad(): 
    test_image_embeddings = model.get_image_embeddings(inputs.pixel_values).squeeze()

# Cosine similarity 
C, h, w = test_image_embeddings.shape 
test_feat = test_image_embeddings / test_image_embeddings.norm(dim=0, keepdim=True) 
test_feat = test_feat.reshape(C, h * w) 
sim = target_feat @ test_feat 
sim = sim.reshape(1, 1, h, w) 
sim = F.interpolate(sim, scale_factor=4, mode="bilinear")

sim = processor.post_process_masks(sim.unsqueeze(1), original_sizes=inputs["original_sizes"].tolist(), 
                                   reshaped_input_sizes=inputs["reshaped_input_sizes"].tolist(), binarize=False) 
sim = sim[0].squeeze()

# Positive location prior 
topk_xy, topk_label = point_selection(sim, topk=1) 
print("Topk_xy:", topk_xy) 
print("Topk_label:", topk_label)

inputs = processor(test_image, input_points=[topk_xy.tolist()], input_labels=[topk_label.tolist()], 
                   return_tensors="pt").to(device) 
for k,v in inputs.items(): 
    print(k,v.shape)

#%% perform cascaded prediction

# First-step prediction 
with torch.no_grad(): 
    outputs = model( input_points=inputs.input_points, input_labels=inputs.input_labels, 
                    image_embeddings=test_image_embeddings.unsqueeze(0), multimask_output=True)

logits = outputs.pred_masks[0].squeeze(0).detach().cpu().numpy() 
logits = logits * weights_np[..., None] 
logit = logits.sum(0)

# Weighted sum three-scale masks 
logits_high = postprocess_masks(masks=outputs.pred_masks.squeeze(1), input_size=inputs.reshaped_input_sizes[0].tolist(), 
                                original_size=inputs.original_sizes[0].tolist()) 
logits_high = logits_high[0] * weights.unsqueeze(-1) 
logit_high = logits_high.sum(0) 
mask = (logit_high > 0).detach().cpu().numpy()

y, x = np.nonzero(mask) 
x_min = x.min() 
x_max = x.max() 
y_min = y.min() 
y_max = y.max() 
input_box = [[x_min, y_min, x_max, y_max]] 
print(input_box)

input_boxes = processor(test_image, input_boxes=[input_box], return_tensors="pt").input_boxes.to(device) 
# Cascaded Post-refinement-1 
with torch.no_grad(): 
    outputs_1 = model( input_points=inputs.input_points, input_labels=inputs.input_labels, 
                      input_boxes=input_boxes, input_masks=torch.tensor(logit[None, None, :, :], device=device), 
                      image_embeddings=test_image_embeddings.unsqueeze(0), multimask_output=True)
    
# Cascaded Post-refinement-2 
masks = processor.image_processor.post_process_masks(outputs_1.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0].squeeze().numpy() 
best_idx = torch.argmax(outputs_1.iou_scores).item() 
y, x = np.nonzero(masks[best_idx])
x_min = x.min() 
x_max = x.max() 
y_min = y.min() 
y_max = y.max() 
input_box = [[[x_min, y_min, x_max, y_max]]] 
print("Input boxes:", input_boxes) 
input_boxes = processor(test_image, input_boxes=[input_box], return_tensors="pt").input_boxes.to(device) 
final_outputs = model( input_points=inputs.input_points, input_labels=inputs.input_labels, input_boxes=input_boxes, 
                      input_masks=outputs_1.pred_masks.squeeze(1)[:,best_idx: best_idx + 1, :, :], 
                      image_embeddings=test_image_embeddings.unsqueeze(0), multimask_output=True)
#final_outputs.pred_masks.shape
#final_outputs.iou_scores
best_idx = torch.argmax(final_outputs.iou_scores).item()
masks = processor.image_processor.post_process_masks(final_outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), 
                                                     inputs["reshaped_input_sizes"].cpu())[0].squeeze().numpy()

#%% Visualize 
def show_mask(mask, ax, random_color=False): 
    if random_color: color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) 
    else: color = np.array([255, 0, 0, 0.6]) 
    h, w = mask.shape[-2:] 
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) 
    ax.imshow(mask_image) 
fig, axes = plt.subplots() 
axes.imshow(np.array(test_image)) 
show_mask(masks[best_idx], axes) 
axes.title.set_text(f"Predicted mask") 
axes.axis("off")
# %% What would the output looks like without training?
model_raw = SamModel.from_pretrained("facebook/sam-vit-huge")
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

input_points = [[[151, 122]]]  # prompt
inputs = processor(test_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
masks_raw = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
# For plot showing
seg_2 = masks_raw[0][0]
scores = outputs.iou_scores
input_point = np.array(input_points[0])
input_label = np.array([1])
scores = np.array(scores.cpu()[0][0])

for i, (mask, score) in enumerate(zip(seg_2, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(test_image)
    show_points(input_point, input_label, plt.gca())
    show_mask(mask, plt.gca())
    plt.axis('off')
    plt.show()  


# %%
