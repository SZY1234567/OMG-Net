import lightning as L
import torch
import numpy as np
from segment_anything_fast import sam_model_fast_registry 
from segment_anything_fast.utils.amg import (
    build_point_grid,
    MaskData,
    batch_iterator,
    calculate_stability_score,
    batched_mask_to_box,
)

import matplotlib.pyplot as plt
from segment_anything_fast.utils import amg
from torchvision.ops.boxes import batched_nms, box_area

from segment_anything_fast.utils.transforms import ResizeLongestSide  ## @@@
import random

import torch
import torch.nn.functional as F


import cucim.skimage as cs

import cupy as cp

class MaskGenerator(L.LightningModule):

    def __init__(self, config, SAM_CHECKPOINT):
        super().__init__()
        self.sam_model = sam_model_fast_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    
        self.config = config
        self.SAM_CHECKPOINT = SAM_CHECKPOINT

        self.n_points = self.config["SAM_MODEL"]["points_per_side"]
        self.point_grids = build_point_grid(self.n_points) * 1024  # @@@@

        self.transform = ResizeLongestSide(1024)
        self.Input_Size = torch.as_tensor(
            self.config["CLASSIFICATION_MODEL"]["Input_Size"],
            dtype=torch.int,
            device=self.device,
        )

        # def setup(self, stage):
            
        #     in_points = torch.as_tensor(self.point_grids, device=self.device)
        #     in_labels = torch.ones(
        #         in_points.shape[0], dtype=torch.int, device=in_points.device
        #     )
        #     in_points = in_points[:, None, :]
        #     in_labels = in_labels[:, None]
        #     self.sparse_embeddings, self.dense_embeddings = self.sam_model.prompt_encoder(
        #             points=(in_points, in_labels), boxes=None, masks=None
        #         )
        
    def forward(self, patches):
        input_image_torch = torch.as_tensor(patches[0], device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_images = self.sam_model.preprocess(input_image_torch)
        self.features = self.sam_model.image_encoder(input_images)
        del input_images
        batch_size = self.config["SAM_MODEL"]["Points_Batch_Size"]
        data = MaskData()
        num_pts_found = 0

        for (points,) in batch_iterator(batch_size, self.point_grids):
            batch_data = MaskData()
            in_points = torch.as_tensor(points, device=self.device)
            in_labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            in_points = in_points[:, None, :]
            in_labels = in_labels[:, None]
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=(in_points, in_labels), boxes=None, masks=None
            )
            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,  ## @@
            )
            batch_data["masks"] = self.sam_model.postprocess_masks(
                low_res_masks, [1024, 1024], [1024, 1024]
            )  

            del low_res_masks
            batch_data["masks"] = batch_data["masks"].flatten(0, 1)
            batch_data["iou_predictions"] = iou_predictions.flatten(0, 1)

            ## Remove based on IOU
            keep_by_iou = (
                batch_data["iou_predictions"]
                > self.config["SAM_MODEL"]["pred_iou_thresh"]
            )
            batch_data.filter(keep_by_iou)

            ##Remove based on Stability
            stability_scores = calculate_stability_score(
                batch_data["masks"], 0, 1.0
            ).to(device=self.device)
            keep_by_stability = (
                stability_scores > self.config["SAM_MODEL"]["stability_score_thresh"]
            )
            batch_data.filter(keep_by_stability)

            ## Remove per area size

            batch_data["masks"] = batch_data["masks"] > 0
            batch_data["boxes"] = batched_mask_to_box(batch_data["masks"]).to(
                device=self.device
            )

            areas               = box_area(batch_data["boxes"]).to(device=self.device)
            keep_per_min = areas > self.config['SAM_MODEL']["min_mask_region_area"]
            keep_per_max = areas < self.config['SAM_MODEL']["max_mask_region_area"]
            keep_per_area = keep_per_min & keep_per_max
            batch_data.filter(keep_per_area)
            
            if batch_data["boxes"].size()[0] == 0:  ## Remove the case where there is no masks
                continue

            # Padding last dimension to avoid masks errors on the edge

            batch_data["padded_masks"] = torch.nn.functional.pad(
                batch_data["masks"], (32, 32, 32, 32), "constant", 0
            )
            centers = []
            cropped_masks = []
            for i, mask in enumerate(batch_data["padded_masks"]):
                bbox = batch_data["boxes"][i] + 32  ## To account for the padding

                center = torch.zeros((2), device=self.device, dtype=torch.int)
                center[0] = (bbox[1] + bbox[3]) / 2
                center[1] = (bbox[0] + bbox[2]) / 2

                cropped_mask = mask[
                    int(center[0] - self.Input_Size[0] / 2) : int(
                        center[0] + self.Input_Size[0] / 2
                    ),
                    int(center[1] - self.Input_Size[1] / 2) : int(
                        center[1] + self.Input_Size[1] / 2
                    ),
                ]

                cropped_masks.append(cropped_mask)

                center[0] -= 32  ## Patches frame of reference
                center[1] -= 32
                centers.append(center)

            batch_data["centers"] = centers
            batch_data["cropped_masks"] = cropped_masks
            num_pts_found += len(centers)
            # del batch_data["masks"]
            del batch_data["padded_masks"]
            data.cat(batch_data)
        if num_pts_found == 0:
            return None
        else:

            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["iou_predictions"],
                torch.zeros_like(data["boxes"][:, 0]),
                iou_threshold=self.config["SAM_MODEL"]["box_nms_thresh"],
            )
            data.filter(keep_by_nms)
            # self.show_anns(data, patches)

            return data

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        patches, ids = batch
        data = self.forward(patches)

        if data is not None:
            cropped_masks = torch.stack(data["cropped_masks"], dim=0)
            centers = torch.stack(data["centers"], dim=0)

            ids = ids.repeat_interleave(cropped_masks.size()[0])
            ## Collapse masks per patch in mask over all patches, make the same for centers
            cropped_masks = cropped_masks.view(
                -1, cropped_masks.shape[-2], cropped_masks.shape[-1]
            )
            centers = centers.view(-1, centers.shape[-1])

            return cropped_masks, centers, ids  # data

        else:

            return torch.empty((0, 64, 64)), torch.empty((0, 2)), torch.empty(0)

    def show_anns(self, data, patches):

        masks, h_stain, centroids = self.find_nucleus_on_H(patches)
        fig, axs = plt.subplots(1, 4, figsize=(12, 4), sharex=True, sharey=True)
        axs[2].imshow(h_stain)
        axs[3].imshow(patches.cpu().squeeze())
        axs[3].scatter(centroids[:, 1], centroids[:, 0], c='red', s=10)
        axs[0].imshow(patches.cpu().squeeze())
        axs[1].imshow(patches.cpu().squeeze())
        # Plot masks
        unique_labels = np.unique(masks)

        mask_img = np.ones((data["masks"][0].shape[0], data["masks"][0].shape[1], 4))
        mask_img[:, :, 3] = 0
        for label in unique_labels[1:]:  # Skip background (0)
            mask = masks == label
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            mask_img[mask] = color_mask

        axs[3].imshow(mask_img)
        if len(data["masks"]) == 0:
            return
        axs[1].set_autoscale_on(False)

        img = np.ones((data["masks"][0].shape[0], data["masks"][0].shape[1], 4))
        img[:, :, 3] = 0
        for mask in data["masks"]:

            m = mask.cpu().numpy()
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        axs[1].imshow(img)

        plt.savefig(f"figure_{random.randint(1, 100000)}.png")

        plt.clf()
        plt.cla()
        plt.close()

    def stain_decomposition(self, patches):
        patches = cp.asarray(patches)
        results = cs.color.separate_stains(patches, cs.color.hed_from_rgb)
        h, e = results[:, :, :, 0], results[:, :, :, 1]
        return h.squeeze(), e.squeeze()

    def identify_nuclei(self, h_stain):
        thresh = cs.filters.threshold_otsu(h_stain)
        binary = (h_stain >thresh).astype(np.uint8)
        opened = cs.morphology.binary_opening(binary)
        return cs.measure.label(opened)
    
    def compute_centroids(self,mask):
        labels = cp.unique(mask)[1:]  # Exclude background (0)
        y, x = cp.indices(mask.shape)
        centroids = cp.array([(cp.mean(y[mask == label]), cp.mean(x[mask == label])) for label in labels])
        return centroids

    def find_nucleus_on_H(self, patches):
        H, _ = self.stain_decomposition(patches)
        masks = self.identify_nuclei(H)
        centroids = self.compute_centroids(masks)
        return torch.as_tensor(masks), torch.as_tensor(H), torch.as_tensor(centroids)

