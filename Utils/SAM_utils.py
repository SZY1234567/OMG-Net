import os
import pandas as pd
import numpy as np
import tifffile
from torchvision.ops import nms
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pycocotools import mask as mask_utils
import cv2

def get_bbox_from_mask(mask):
    pos = np.where(mask != 0)
    if pos[0].shape[0] == 0:
        return np.zeros((0, 4))
    else:
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

def start_points(size, split_size, overlap_ratio=0.2):
    points = [0]
    stride = int(split_size * (1-overlap_ratio))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size: break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def patching(image, dim=(500, 500), overlap_ratio=0.2):#image : (W,H,C)
    im_w, im_h = image.shape[0], image.shape[1]
    starts_x = start_points(im_w, dim[0], overlap_ratio=overlap_ratio)
    starts_y = start_points(im_h, dim[1], overlap_ratio=overlap_ratio)
    patch_dict = {}
    for i in starts_x:
        for j in starts_y:
            coords = [i, j]
            patch = image[coords[0]:coords[0] + dim[0], coords[1]:coords[1] + dim[1], :]
            padded = np.zeros_like(image[:dim[0], :dim[1], :], dtype='uint8')
            padded[:patch.shape[0], :patch.shape[1], :] = patch
            patch_dict[str(coords)] = padded
    return patch_dict

def plot_on_ax(ax, img, title):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=16)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_anns(ax, anns):
    if len(anns) == 0: return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.75)))

def show_masks(ax, masks, color_mask=None, alpha=0.5):
    if len(masks) == 0: return
    ax.set_autoscale_on(False)
    for m in masks:
        img = np.ones((m.shape[0], m.shape[1], 3))
        if color_mask is None: color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*alpha)))

def mask2poly(coords, ann):
    m = mask_utils.encode(np.asfortranarray(ann['segmentation']))
    mask = mask_utils.decode(m)
    cc, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cc = [np.squeeze(c) for c in cc]  # Convert contours to the correct shape
    cc = [np.atleast_2d(c) for c in cc]
    cc = cc[0].tolist()
    cc = [[int(c[0] + coords[1]), int(c[1] + coords[0])] for c in cc]
    cc.append(cc[0])
    return cc

def post_precessing_patch(config, coords, image_shape, masks, edge_size = 10):
    selected_masks = []
    masks = [mask for mask in masks if mask['area'] < config['SAM_MODEL']['max_mask_region_area']]
    for mask in masks:
        pos = np.where(mask['segmentation'])
        xmin_close = np.any(pos[0] < edge_size)
        ymin_close = np.any(pos[1] < edge_size)
        xmax_close = np.any(abs(pos[0] - config['DATA']['Patch_Size'][0]) < edge_size)
        ymax_close = np.any(abs(pos[1] - config['DATA']['Patch_Size'][1]) < edge_size)

        if coords[0] == 0:
            if coords[1] == 0:
                selection = xmax_close | ymax_close
            elif coords[1] == (image_shape[1] - config['DATA']['Patch_Size'][1]):
                selection = xmax_close | ymin_close
            else:
                selection = xmax_close | ymax_close | ymin_close

        elif coords[0] == (image_shape[0] - config['DATA']['Patch_Size'][0]):
            if coords[1] == 0:
                selection = xmin_close | ymax_close
            elif coords[1] == (image_shape[1] - config['DATA']['Patch_Size'][1]):
                selection = xmin_close | ymin_close
            else:
                selection = xmin_close | ymax_close | ymin_close
        else:
            if coords[1] == 0:
                selection = xmin_close | xmax_close | ymax_close
            elif coords[1] == (image_shape[1] - config['DATA']['Patch_Size'][1]):
                selection = xmin_close | xmax_close | ymin_close
            else:
                selection = xmin_close | xmax_close | ymax_close | ymin_close

        if ~selection: selected_masks.append(mask)

    print("Number of masks after post-processing: {}/{}".format(len(selected_masks), len(masks)))
    return selected_masks

def post_precessing_image(config, masks):
    print("Total number of masks in: {}".format(len(masks)))
    bboxes = np.array([ann['bbox'] for ann in masks])
    scores = np.array([ann['predicted_iou'] for ann in masks])
    return nms(boxes=torch.tensor(bboxes, dtype=torch.float32),
               scores=torch.tensor(scores, dtype=torch.float32),
               iou_threshold=config['SAM_MODEL']['box_nms_thresh']).tolist()

def masks_nms(masks, iou_threshold=0.01):
    bboxes = np.array([ann['bbox'] for ann in masks])
    scores = np.array([ann['predicted_iou'] for ann in masks])
    keep = nms(boxes=torch.tensor(bboxes, dtype=torch.float32),
               scores=torch.tensor(scores, dtype=torch.float32),
               iou_threshold=iou_threshold).tolist()
    print("Number of masks after NMS: {}".format(len(keep), len(masks)))
    return [mask for j, mask in enumerate(masks) if j in keep]

def combine_masks(masks):
    instance_map = np.zeros_like(masks[0]).astype("uint8")
    for i, mask in enumerate(masks):
        pos = np.where(mask != 0)
        instance_map[pos[0], pos[1]] = i + 1
    return instance_map

def remove_edge_masks(image_shape, masks, edge_size = 5):
    selected_masks = []
    for mask in masks:
        pos = np.where(mask)
        xmin_close = np.any(pos[0] < edge_size)
        ymin_close = np.any(pos[1] < edge_size)
        xmax_close = np.any(abs(pos[0] - image_shape[0]) < edge_size)
        ymax_close = np.any(abs(pos[1] - image_shape[1]) < edge_size)
        selection = xmin_close | xmax_close | ymax_close | ymin_close
        if ~selection: selected_masks.append(mask)

    return selected_masks


