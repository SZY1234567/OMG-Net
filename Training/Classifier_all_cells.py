import toml
import sys
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import lightning as L
import torch
torch.set_float32_matmul_precision("medium")

import albumentations as A
import torchvision.transforms as T
from Utils import ColourAugment
from Models.SAM_Classifier import Classifier
from Dataloader.SAM import DataModule
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import nrrd
import time
from datetime import datetime
from Utils.SAM_utils import *
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.ops import nms
from torchvision.transforms import v2

def load_config(config_file):
    return toml.load(config_file)

def get_logger(config, timestamp):
    return TensorBoardLogger(os.path.join(config['CHECKPOINT']['logger_folder'],
                                          config['CHECKPOINT']['model_name'],
                                          config['BASEMODEL']['Backbone'],
                                          ),
                             name="Mask_Input_" + str(config['BASEMODEL']['Mask_Input']), version=timestamp)
def get_callbacks(config, save_dir):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath     = save_dir,
        monitor     = config['CHECKPOINT']['Monitor'],
        filename    = config['CHECKPOINT']['filename'],
        save_top_k  = config['CHECKPOINT']['save_top_k'],
        mode        = config['CHECKPOINT']['Mode'])

    return [lr_monitor, checkpoint_callback]

def get_transforms(config):
    augmentation = A.Compose([
        A.RandomCrop(width=config['DATA']['Input_Size'][0], height=config['DATA']['Input_Size'][1]),
        A.HorizontalFlip(p=config['AUGMENTATION']['horizontalflip']),
        A.RandomBrightnessContrast(p=config['AUGMENTATION']['randombrightnesscontrast']),
    ])

    train_normalization = T.Compose([
        T.ToTensor(),
        ColourAugment.ColourAugment(sigma=config['AUGMENTATION']['Colour_Sigma'],
                                    mode=config['AUGMENTATION']['Colour_Mode']),
        v2.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_normalization = T.Compose([
        T.ToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return augmentation, train_normalization, val_normalization

def get_SAM_masks(config, df, csv_file):

    if not os.path.isdir(os.path.join(config['DATA']['output_path'], "masks")):
        os.mkdir(os.path.join(config['DATA']['output_path'], "masks"))
    if not os.path.isdir(os.path.join(config['DATA']['output_path'], "figures")):
        os.mkdir(os.path.join(config['DATA']['output_path'], "figures"))

    sam_checkpoint = "/home/dgs2/Software/SAM/SAMCheckPoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=config['SAM_MODEL']['points_per_side'],
        points_per_batch=config['SAM_MODEL']['points_per_batch'],
        pred_iou_thresh=config['SAM_MODEL']['pred_iou_thresh'],
        stability_score_thresh=config['SAM_MODEL']['stability_score_thresh'],
        box_nms_thresh=config['SAM_MODEL']['box_nms_thresh'],
        min_mask_region_area=config['SAM_MODEL']['min_mask_region_area'],
    )

    custom_field_map = {
                'SVS_ID': 'string',
                'top_left': 'int list',
                'center': 'int list',
                'dim': 'int list',
                'vis_level': 'int',
                'diagnosis': 'string',
                'annotation_label': 'string',
                'mask': 'double matrix'}

    masks_datasets = []

    for i in range(len(df)):
        image_name = df['nrrd_file'][i].split(".")[0]
        label = df['class'][i]
        nrrd_data = os.path.join(config['DATA']['nrrd_path'], df['nrrd_file'][i])
        img, header = nrrd.read(nrrd_data, custom_field_map=custom_field_map)
        img = img[:, :, :3]
        msk = np.array(header['mask'])

        if os.path.isfile(os.path.join(config['DATA']['output_path'], "masks", "masks_{}.npy".format(image_name))):
            instance_map = np.load(os.path.join(config['DATA']['output_path'], "masks", "masks_{}.npy".format(image_name)), allow_pickle=True)
        else:
            print("Generating SAM masks on Image No.{}/{}: {}".format(i+1, len(df), image_name) +  "-" * 100)
            anns = mask_generator.generate(img)
            anns = [ann for ann in anns if ann['area'] < config['SAM_MODEL']['max_mask_region_area']]
            anns = [ann for ann in anns if ann['area'] > config['SAM_MODEL']['min_mask_region_area']]

            masks = [np.array(ann['segmentation'], dtype='float32') for ann in anns]
            masks.append(msk)
            bboxes = [get_bbox_from_mask(mask) for mask in masks]
            scores = [0.1]*(len(bboxes)-1)
            scores.append(1.0)

            keep = sorted(nms(boxes=torch.tensor(np.array(bboxes), dtype=torch.float32),
                              scores=torch.tensor(np.array(scores), dtype=torch.float32),
                              iou_threshold=config['SAM_MODEL']['box_nms_thresh']).tolist())

            assert keep[-1] == (len(masks)-1)
            masks = [mask for j, mask in enumerate(masks) if j in keep]
            masks = remove_edge_masks(img.shape, masks, edge_size = 2)

            if len(masks) == 0:
                print("No masks found in {}".format(image_name))
                continue

            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            plot_on_ax(axs[0], img, "image")
            plot_on_ax(axs[1], img, "masks")
            show_masks(axs[1], masks[:-1])
            show_masks(axs[1], [masks[-1]], [0.1,0.9,0.1], alpha=0.9)
            fig.suptitle(image_name + "Class: {}".format(label))
            plt.tight_layout()
            plt.savefig(os.path.join(config['DATA']['output_path'], "figures", '{}.png'.format(image_name)))
            plt.close()

            instance_map = combine_masks(masks)
            np.save(os.path.join(config['DATA']['output_path'], "masks", "masks_{}.npy".format(image_name)), instance_map)

        df_masks = pd.DataFrame()
        df_masks['mask_id'] = range(1, instance_map.max() + 1)
        df_masks['image_id'] = df['image_id'][i]
        df_masks['nrrd_file'] = df['nrrd_file'][i]
        df_masks['masks_file'] = "masks_{}.npy".format(image_name)
        df_masks['class'] = [0] * (instance_map.max() - 1) + [label]

        for id in df_masks['mask_id'].to_list():
            if sum(sum(instance_map == id) != 0) == 0:
                print("Deleting {}".format(df_masks[df_masks['mask_id'] == id]))
                df_masks.drop(df_masks[df_masks['mask_id'] == id].index, inplace=True)

        df_masks.reset_index(drop=True, inplace=True)
        masks_datasets.append(df_masks)
        print(f"masks_dataset of Image No.{i + 1}/{len(df)}: {image_name} added" + "-" * 100)

    masks_dataset = pd.concat(masks_datasets, axis=0).reset_index(drop=True)
    masks_dataset.to_csv(os.path.join(config['DATA']['output_path'], csv_file), index=False)
    print(masks_dataset)
    print("{} saved".format(os.path.join(config['DATA']['output_path'], csv_file)))

def get_datasets(config):
    df = pd.read_csv(config['DATA']['Dataframe'])
    df = df[(df['image_quality'] == 1) & (df['class'] != 999)]
    df = df[df['species'].isin(config['DATA']['species'])]
    df = df[df['source'].isin(config['DATA']['datasets'])]
    df.reset_index(drop=True, inplace=True)
    print(df)

    image_ids = df['image_id'].unique()

    if not os.path.isdir(config['DATA']['output_path']):
        os.mkdir(config['DATA']['output_path'])

    if not os.path.isfile(os.path.join(config['DATA']['output_path'], "masks_dataset.csv")):
        get_SAM_masks(config, df, csv_file="masks_dataset.csv")
        
    df = pd.read_csv(os.path.join(config['DATA']['output_path'], "masks_dataset.csv"))
    df = df[df['image_id'].isin(image_ids)]
    print(df)

    masks_dataset_test = df[df['image_id'].isin(config['DATA']['filenames_test'])].reset_index(drop=True)
    df_train_val = df[~df['image_id'].isin(config['DATA']['filenames_test'])].reset_index(drop=True)

    filenames = list(df_train_val['image_id'].unique())
    train_idx, val_idx = train_test_split(filenames, test_size=config['DATA']['val_size'])

    masks_dataset_train = df[df['image_id'].isin(train_idx)]
    masks_dataset_train.reset_index(drop=True, inplace=True)

    masks_dataset_val = df[df['image_id'].isin(val_idx)]
    masks_dataset_val.reset_index(drop=True, inplace=True)

    def balancing(df, target_label):
        N_min = min([len(group) for label, group in df.groupby(target_label)])
        return (df.
                groupby(target_label).
                apply(lambda group: group.sample(N_min, replace=False))
                .reset_index(drop=True))

    if config['DATA']['balancing']:
        masks_dataset_train = balancing(masks_dataset_train, 'class')
        masks_dataset_val = balancing(masks_dataset_val, 'class')

    N = len(masks_dataset_train) + len(masks_dataset_val) + len(masks_dataset_test)

    print('Training Size: {}/{}({}) Positive Rate: {}'.format(len(masks_dataset_train), N, len(masks_dataset_train) / N,
                                                              list(masks_dataset_train['class'].value_counts(normalize=True))[1]))
    print('Validation Size: {}/{}({}) Positive Rate: {}'.format(len(masks_dataset_val), N, len(masks_dataset_val) / N,
                                                                list(masks_dataset_val['class'].value_counts(normalize=True))[1]))
    print('Testing Size: {}/{}({}) Positive Rate: {}'.format(len(masks_dataset_test), N, len(masks_dataset_test) / N,
                                                             list(masks_dataset_test['class'].value_counts(normalize=True))[1]))

    return masks_dataset_train, masks_dataset_val, masks_dataset_test

def main(config_file):
    start = time.time()
    config = toml.load(config_file)
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    augmentation, train_normalization, val_normalization = get_transforms(config)
    masks_dataset_train, masks_dataset_val, masks_dataset_test = get_datasets(config)
    print("-" * 50 + "Time Elapsed: {}".format(time.time() - start) + "-" * 50)

    data = DataModule(df_train = masks_dataset_train,
                      df_val = masks_dataset_val,
                      df_test = masks_dataset_test,
                      config = config,
                      train_normalization=train_normalization,
                      val_normalization=val_normalization,
                      augmentation=augmentation,
                      inference=False,
                      )

    logger = get_logger(config, timestamp)
    print(logger.log_dir)
    callbacks = get_callbacks(config, logger.log_dir)

    L.seed_everything(config['BASEMODEL']['Random_Seed'], workers=True)

    model = Classifier(config)
    trainer = L.Trainer(devices=config['BASEMODEL']['GPU_ID'],
                        accelerator="gpu",
                        benchmark=False,
                        max_epochs=config['BASEMODEL']['Max_Epochs'],
                        callbacks=callbacks,
                        logger=logger,
                        )

    trainer.fit(model, data)
    trainer.test(ckpt_path='best', dataloaders=data.test_dataloader())

    print("-" * 50 + "Time Elapsed: {}".format(time.time() - start) + "-" * 50)

if __name__ == "__main__":
    main(sys.argv[1])