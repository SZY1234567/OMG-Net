import sys, os
import multiprocessing as mp
sys.path.append(os.getcwd())
from Dataloader.Dataloader import DataGenerator
from Dataloader.SAM import DataGeneratorInf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import torch
import time
import lightning as L
import pandas as pd
from Models.Preprocessing.depreciated.ConvNet import ConvNet as preprocessing_model # old model - use for now.
from Models.SAM_Classifier import Classifier
from Models.SAM_Masking import MaskGenerator
import numpy as np
from Utils import TileOps

## Globals
n_gpus      = torch.cuda.device_count()
devices     = [0,1,2,3]
num_workers = 20
n_ensemble  = 5

def load_config():
    config = {
        'BASEMODEL': {
            'Image_Type': ".svs",
            'WSIReader': "cuCIM",
            'Input_Size': [64, 64],
            'Patch_Size': [256, 256],
            'Mask_Input': True,
            'Precision': '16-mixed',
            'Vis': [0],
            'Batch_Size_Preprocessing': 128,
            'Batch_Size_Masking': 1,
            'Batch_Size_Classification': 3000,
            'Prob_Tumour_Tresh': 0.85
        },
        'SAM_MODEL': {
            'points_per_side': 32,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.8,
            'box_nms_thresh': 0.1,
            'min_mask_region_area': 36,
            'max_mask_region_area': 3600
        },
    }
    return config


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def RemoveBackground(local_SVS_PATH, config):
    tile_coords_no_background  = TileOps.get_nonbackground_tiles(local_SVS_PATH, config)
    tile_dataset_preprocessing = pd.DataFrame({'coords_x': tile_coords_no_background[:, 0],
                                               'coords_y': tile_coords_no_background[:, 1]})  
    tile_dataset_preprocessing['SVS_PATH'] = local_SVS_PATH

    ## Manually sample
    tile_dataset_preprocessing = tile_dataset_preprocessing.sample(frac=1, random_state=42).reset_index(drop=True) ## Balancing
    tiles_per_gpu = len(tile_dataset_preprocessing) // trainer.world_size
    start_idx     = trainer.global_rank * tiles_per_gpu
    end_idx = start_idx + tiles_per_gpu if trainer.global_rank < trainer.world_size - 1 else len(tile_dataset_preprocessing)
    tile_dataset_preprocessing = tile_dataset_preprocessing[start_idx:end_idx]

    print(trainer.global_rank, start_idx, end_idx)    
    return tile_dataset_preprocessing   

def MaskGeneration(tile_dataset_preprocessing, SAM_CHECKPOINT, config):
    # tile_dataset_preprocessing.reset_index(drop=True, inplace=True)
    mask_transform = v2.Compose([v2.ToImage(),
                                 v2.ToDtype(torch.half, scale=False)])
    model_maskgenerator = MaskGenerator(config, SAM_CHECKPOINT)
    model_maskgenerator.eval()
    model_maskgenerator = freeze_model(model_maskgenerator)
    
    # Create dataloader
    data = DataLoader(DataGenerator(tile_dataset_preprocessing, config, transform = mask_transform),
                      batch_size=config['BASEMODEL']['Batch_Size_Masking'],
                      num_workers=num_workers,
                      pin_memory=False,
                      shuffle=False)

    predictions         = trainer.predict(model_maskgenerator, data)

    cropped_masks       = torch.cat([cropped_mask for cropped_mask, center, idx in predictions], dim=0)##[NMask, H, W]
    centers             = torch.cat([center for cropped_mask, center, idx in predictions], dim=0) ## [NMasks,2]
    indexes             = torch.cat([idx for cropped_mask, center, idx in predictions], dim=0) ## [NMask ]
    
    return cropped_masks, centers, indexes

def CellClassification(trainer, tile_dataset_preprocessing, cropped_masks, centers, indexes, CLASSIFY_CHECKPOINT, config):
    classif_transform = transforms.Compose([transforms.ToTensor(),  
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            transforms.Lambda(lambda x: x.half()),
    ])
    tile_dataset_preprocessing.reset_index(drop=True, inplace=True)
    # Create dataloader
    data = DataLoader(DataGeneratorInf(config, tile_dataset_preprocessing,
                                       masks=cropped_masks,
                                       centers=centers,
                                       indexes = indexes,
                                       transform=classif_transform),
                      batch_size=config['BASEMODEL']['Batch_Size_Classification'],
                      num_workers=num_workers,
                      shuffle=False,
                      pin_memory=False)
    del cropped_masks

    # Load model and freeze layers

    cell_type_predictions_all_models = []
    coords_all_models  = []
    
    for i in range(n_ensemble):
        model_classifier = Classifier.load_from_checkpoint(CLASSIFY_CHECKPOINT[i])
        model_classifier.eval()
        model_classifier = freeze_model(model_classifier)

        predictions                     = trainer.predict(model_classifier, data)
        cell_type_predictions           = torch.cat([prob for prob, coords, idx in predictions], dim=0)
        coords                          = torch.cat([coords for prob, coords, idx in predictions], dim=0)
        indexes                         = torch.cat([idx for prob, coords, idx in predictions], dim=0)
        
        cell_type_predictions           = cell_type_predictions.view(-1, cell_type_predictions.shape[-1])
        coords                          = coords.view(-1, coords.shape[-1])
        indexes                         = indexes.view(-1)
        
        cell_type_predictions           = cell_type_predictions[indexes]
        coords                          = coords[indexes]
        cell_type_predictions_all_models.append(cell_type_predictions)
        coords_all_models.append(coords)
    
    cell_type_predictions = torch.mean(torch.stack(cell_type_predictions_all_models).float(), dim=0)
    coords                = torch.mean(torch.stack(coords_all_models).float(), dim=0)    
        
    return cell_type_predictions, coords

def complete_inference(config, trainer, local_SVS_PATH, local_result, PROCESSING_CHECKPOINT, SAM_CHECKPOINT, CLASSIFY_CHECKPOINT): 
    try:
        print('1. Remove non-background tiles')
        time_start = time.time()
        tile_dataset_preprocessing = RemoveBackground(local_SVS_PATH, config)
        time_end = time.time()
        if trainer.is_global_zero:
            print(f'{len(tile_dataset_preprocessing)} tiles to classify after background removal [{int(time_end - time_start)} seconds]')
    except Exception as e:
        raise RuntimeError("Critical failure during background removal") from e

    if not os.path.exists(f'{local_result[:-4]}-masks.pth'):
        try:   
            print('3. Mask Generation')
            cropped_masks, centers, indexes = MaskGeneration(tile_dataset_preprocessing, SAM_CHECKPOINT, config)
            time_maskgeneration = time.time()
            if trainer.is_global_zero:  
                print(f'Mask generation completed [{int(time_maskgeneration - time_end)} seconds]')
            torch.save({'cropped_masks': cropped_masks, 'centers': centers, 'indexes': indexes}, f'{local_result[:-4]}-masks.pth')
            print(f'Masks saved at {local_result[:-4]}-masks.pth !')
        except Exception as e:
            raise RuntimeError("Critical failure during mask generation") from e
    else:
        print(f'Loading Masks from {local_result[:-4]}-masks.pth ... ')
        masks_file = torch.load(f'{local_result[:-4]}-masks.pth')
        cropped_masks, centers, indexes = masks_file['cropped_masks'], masks_file['centers'], masks_file['indexes']

    try:
        print('4. Classification')
        cell_type_predictions, coords =  CellClassification(trainer, tile_dataset_preprocessing, cropped_masks, centers, indexes, CLASSIFY_CHECKPOINT, config)
    except Exception as e:
        raise RuntimeError("Critical failure during mask classification") from e
        
    masks_dataset             = pd.DataFrame()
    coords = coords.numpy().astype(int)
    masks_dataset['coords_x'] = coords[:, 0]
    masks_dataset['coords_y'] = coords[:, 1]
    masks_dataset['SVS_PATH'] = local_SVS_PATH

    for n, class_label in enumerate(['pred_0', 'pred_1']):
        masks_dataset[class_label] = cell_type_predictions[:, n]

    np.savez(f"{local_result[:-4]}_{trainer.global_rank}.npz", masks=cropped_masks.numpy(), coords=coords)
    
    masks_dataset = masks_dataset[masks_dataset['pred_1']>0.5]
    # ## NMS again - not sure I like it
    # bboxes = np.zeros((len(masks_dataset), 4))
    # bboxes[:, 0] = masks_dataset['coords_x'] - config['BASEMODEL']['Input_Size'][0]/2
    # bboxes[:, 1] = masks_dataset['coords_y'] - config['BASEMODEL']['Input_Size'][1]/2
    # bboxes[:, 2] = masks_dataset['coords_x'] + config['BASEMODEL']['Input_Size'][0]/2
    # bboxes[:, 3] = masks_dataset['coords_y'] + config['BASEMODEL']['Input_Size'][1]/2
    # keep = torchvision.ops.nms(boxes=torch.tensor(np.array(bboxes), dtype=torch.float32),
    #                            scores=torch.tensor(np.array(masks_dataset['pred_1']), dtype=torch.float32),
    #                            iou_threshold=0.1).tolist()

    # masks_dataset = masks_dataset.iloc[keep]
    masks_dataset.to_csv(f"{local_result[:-4]}_{trainer.global_rank}.csv", index=False)
    print(masks_dataset)
    print(f"{local_result[:-4]}_{trainer.global_rank}.csv Saved")
    trainer.strategy.barrier() ##Synchronize all the save

    if trainer.is_global_zero:
        dataset_dict = {}
        npz_dict = {}
        for gpu_id in range(trainer.world_size):
            npz_dict[f"{gpu_id}"] = np.load(f"{local_result[:-4]}_{gpu_id}.npz", allow_pickle=True)
            dataset = pd.read_csv(f"{local_result[:-4]}_{gpu_id}.csv")
            print(f"{local_result[:-4]}_{gpu_id}.csv")
            print(f"{local_result[:-4]}_{gpu_id}.npz")
            if not dataset.empty:
                dataset_dict[f"{gpu_id}"] = dataset  

        masks = np.concatenate([npz_dict[f"{gpu_id}"]['masks'] for gpu_id in range(trainer.world_size)])
        coords = np.concatenate([npz_dict[f"{gpu_id}"]['coords'] for gpu_id in range(trainer.world_size)]).astype(int)
        np.savez(f"{local_result[:-4]}.npz", masks=masks, coords = coords)

        if dataset_dict:
            masks_dataset = pd.concat([value for key, value in dataset_dict.items()], axis=0)
        else:
            masks_dataset = pd.DataFrame()
            
        masks_dataset.to_csv(f"{local_result[:-4]}.csv", index=False)
        print(masks_dataset)
        print(f"Number of cells: {masks.shape[0]}, Number of mitotic figures: {len(masks_dataset)}")
    
    return True

if __name__ == "__main__":

    config          = load_config()

    ## Checkpoints 
    SAM_CHECKPOINT          = "/path/to/sam_vit_h_4b8939.pth"
    CLASSIFY_CHECKPOINT     = [
        "/path/to/MFDetectionV1.ckpt",
        "/path/to/MFDetectionV2.ckpt",
        "/path/to/MFDetectionV3.ckpt",
        "/path/to/MFDetectionV4.ckpt",
        "/path/to/MFDetectionV5.ckpt",
                             ]

    local_SVS_PATH  = sys.argv[1]
    local_result    = f"{os.path.basename(local_SVS_PATH)[:-4]}.csv"

    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')
    trainer = L.Trainer(devices=devices,
                        accelerator="gpu",
                        strategy="ddp",
                        logger=False,
                        precision=config['BASEMODEL']['Precision'],
                        use_distributed_sampler = False,
                        benchmark=False,)

    # Run model
    trainer.strategy.barrier() ## Sync everything to make sure the data is correctly downloaded
    signal = complete_inference(config,
                       trainer,
                       local_SVS_PATH,
                       local_result,
                       PROCESSING_CHECKPOINT,
                       SAM_CHECKPOINT,
                       CLASSIFY_CHECKPOINT)

    if trainer.is_global_zero:     
        if signal:   
            for gpu_id in range(trainer.world_size):
                os.remove(f"{local_result[:-4]}_{gpu_id}.csv")
                os.remove(f"{local_result[:-4]}_{gpu_id}.npz")

