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
import torchvision
from Models.SAM_Classifier import Classifier
from Models.SAM_Masking import MaskGenerator
import numpy as np
import boto3

from Utils import TileOps
from Utils.MitoticIndex import AIMitoticIndex


## Globals
n_gpus      = torch.cuda.device_count()
num_workers = int(.95 * mp.Pool()._processes / n_gpus)
n_ensemble  = 5
def load_config():
    config = {
        'OVERALL': {
            'Precision': '16-mixed',
            'WSIReader': "cuCIM",            
            'Patch_Size': [512,512],
            'Vis': [0],

        },        
        'PREPROCESSING_MODEL': {
            'Batch_Size': 128,
            'Prob_Tumour_Tresh': 0.85
        },
    
        'SAM_MODEL': {
            'Batch_Size': 1,
            'points_per_side': 32,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.8,
            'box_nms_thresh': 0.3,
            'min_mask_region_area': 36,
            'max_mask_region_area': 3600,
            'Points_Batch_Size': 400
        },
        
        'CLASSIFICATION_MODEL': {
            'WSIReader': "cuCIM",
            'Input_Size': [64, 64],
            'Mask_Input': True,
            'Batch_Size': 3000,
        },                
    }
    return config


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def log(inference_row, status):
    #inference_row.job_status = status
    #session.commit()
    return ''

def RemoveBackground(local_SVS_PATH, config):
    tile_coords_no_background  = TileOps.get_nonbackground_tiles(local_SVS_PATH, config)
    tile_dataset_preprocessing = pd.DataFrame({'coords_x': tile_coords_no_background[:, 0],
                                               'coords_y': tile_coords_no_background[:, 1]})  
    tile_dataset_preprocessing['SVS_PATH'] = local_SVS_PATH

    ## Manually sample
    tile_dataset_preprocessing = tile_dataset_preprocessing.sample(frac=1, random_state=42).reset_index(drop=True) ## Balancing
    tiles_per_gpu = len(tile_dataset_preprocessing) // trainer.world_size
    start_idx     = trainer.global_rank * tiles_per_gpu
    end_idx = start_idx + tiles_per_gpu if trainer.global_rank < trainer.world_size - 1 else len(tile_coords_no_background)    
    tile_dataset_preprocessing = tile_dataset_preprocessing[start_idx:end_idx]
    return tile_dataset_preprocessing   
  
def MaskGeneration(tile_dataset_preprocessing, SAM_CHECKPOINT, config):
    mask_transform = v2.Compose([
        v2.ToPILImage(),
        v2.Resize((1024, 1024)),
        v2.Lambda(lambda x: np.array(x))
    ])
    model_maskgenerator = MaskGenerator(config, SAM_CHECKPOINT)
    model_maskgenerator.eval()
    model_maskgenerator = freeze_model(model_maskgenerator)
    
    # Create dataloader
    data = DataLoader(DataGenerator(tile_dataset_preprocessing, config, transform = mask_transform),
                      batch_size=config['SAM_MODEL']['Batch_Size'],
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

    # Create dataloader
    data = DataLoader(DataGeneratorInf(config,
                                       tile_dataset_preprocessing,
                                       masks=cropped_masks,
                                       centers=centers,
                                       indexes = indexes,
                                       transform=classif_transform),
                      batch_size=config['CLASSIFICATION_MODEL']['Batch_Size'],
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
        print(cell_type_predictions)
    
    cell_type_predictions = torch.mean(torch.stack(cell_type_predictions_all_models).float(), dim=0)
    coords                = torch.mean(torch.stack(coords_all_models).float(), dim=0)    
        
    return cell_type_predictions, coords

def complete_inference(config, trainer, local_SVS_PATH, PROCESSING_CHECKPOINT, SAM_CHECKPOINT, CLASSIFY_CHECKPOINT, inference_row):
    try:
        print('1. Remove non-background tiles')
        time_start = time.time()
        tile_dataset_preprocessing = RemoveBackground(local_SVS_PATH, config)
        time_end = time.time()
        if trainer.is_global_zero:
            print(f'{len(tile_dataset_preprocessing)} tiles to classify after background removal [{int(time_end - time_start)} seconds]')
    except Exception as e:
        log(inference_row, f"Error during background removal")
        raise RuntimeError("Critical failure during background removal") from e

    try:   
        print('2. Mask Generation')
        log(inference_row,'RUNNING - Finding Cells')
        cropped_masks, centers, indexes = MaskGeneration(tile_dataset_preprocessing, SAM_CHECKPOINT, config)
        time_maskgeneration = time.time()
        if trainer.is_global_zero:  
            print(f'Mask generation completed [{int(time_maskgeneration - time_start)} seconds]')
    except Exception as e:
        log(inference_row,f"Error during mask generation")
        raise RuntimeError("Critical failure during mask generation") from e

    try:
        print('3. Classification')
        log(inference_row,'RUNNING - Classifying Cells')

        cell_type_predictions, coords =  CellClassification(trainer, tile_dataset_preprocessing, cropped_masks, centers, indexes, CLASSIFY_CHECKPOINT, config)
    except Exception as e:
        log(inference_row,f"Error during classification")
        raise RuntimeError("Critical failure during mask classification") from e
        
        
    masks_dataset             = pd.DataFrame()
    masks_dataset['coords_x'] = coords[:, 0]
    masks_dataset['coords_y'] = coords[:, 1]
    masks_dataset['SVS_PATH'] = local_SVS_PATH

    for n, class_label in enumerate(['pred_0', 'pred_1']):
        masks_dataset[class_label] = cell_type_predictions[:, n]
    print(masks_dataset)
    masks_dataset = masks_dataset[masks_dataset['pred_1']>0.5]
    print("after", masks_dataset)
    ## NMS again - not sure I like it
    bboxes = np.zeros((len(masks_dataset), 4))
    bboxes[:, 0] = masks_dataset['coords_x'] - 32
    bboxes[:, 1] = masks_dataset['coords_y'] - 32
    bboxes[:, 2] = masks_dataset['coords_x'] + 32
    bboxes[:, 3] = masks_dataset['coords_y'] + 32
    keep = torchvision.ops.nms(boxes=torch.tensor(np.array(bboxes), dtype=torch.float32),
                               scores=torch.tensor(np.array(masks_dataset['pred_1']), dtype=torch.float32),
                               iou_threshold=0.1).tolist()

    masks_dataset = masks_dataset.iloc[keep]
    masks_dataset.to_csv(f"{local_SVS_PATH[:-4]}_{trainer.global_rank}.csv", index=False)
    trainer.strategy.barrier() ##Synchronize all the save

    ### Mitotic Index
    if trainer.is_global_zero:
        time_classification = time.time()
        print(f'Tissue type classification completed [{int(time_classification - time_maskgeneration)} seconds]')

        dataset_dict = {}
        for gpu_id in range(trainer.world_size):
            dataset = pd.read_csv(f"{local_SVS_PATH[:-4]}_{gpu_id}.csv")
            if not dataset.empty:
                dataset_dict[f"{gpu_id}"] = dataset    
        
        if dataset_dict: ## Found mitoses 
            masks_dataset = pd.concat([value for key, value in dataset_dict.items()], axis=0)
            masks_dataset.to_csv(local_SVS_PATH[:-4] + ".csv", index=False)
            coord, MI = AIMitoticIndex(masks_dataset)
            return coord,MI
        else: ## Write empty results
            masks_dataset = pd.DataFrame()
            masks_dataset.to_csv(local_SVS_PATH[:-4] + ".csv", index=False)
            return [0,0],0

    else:
        return [0,0], 0

if __name__ == "__main__":

    config          = load_config()
    branch          = os.getenv('BRANCH')

    ## Checkpoints 
    s3_bucket_ckpt          = os.getenv('S3_Bucket_CKPT')
    s3_object_preprocessing = os.getenv('Preprocessing_CKPT')
    s3_sam_mask_generator   = os.getenv('SAM_CKPT')
    s3_object_classifier    = [os.getenv('Classification_CKPT_'+str(i)) for i in range(n_ensemble)]

    local_directory         = './'
    PROCESSING_CHECKPOINT   = os.path.join(local_directory, s3_object_preprocessing)
    SAM_CHECKPOINT          = os.path.join(local_directory, s3_sam_mask_generator)

    CLASSIFY_CHECKPOINT     = [os.path.join(local_directory, s3_object_classifier[i]) for i in range(n_ensemble)]
 
    s3_bucket_slide = 'histopathology-slides'
    slideKey        = os.getenv('slideKey')
    identityID      = os.getenv('identityID')
    file_name       = 'slide.tif'
    file_dir        = os.path.join('protected', identityID, slideKey)

    local_SVS_PATH  = os.path.join(local_directory, file_name)
    local_result    = local_SVS_PATH[:-4]+".csv"
    s3_object_key   = os.path.join(file_dir, file_name)        
    s3_result_key   = os.path.join(file_dir, 'mitoses_result.csv')

    inference_row   = 0
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')
    trainer = L.Trainer(devices=n_gpus,
                        accelerator="gpu",
                        strategy="ddp",
                        logger=False,
                        precision=config['OVERALL']['Precision'],
                        use_distributed_sampler = False,
                        benchmark=False)

    if trainer.is_global_zero:
        # Local path in your container
        os.makedirs(local_directory, exist_ok = True)        

        ## Download all relevant CKPT and slide
        s3_client = boto3.client('s3', region_name='eu-west-2')
        s3_client.download_file(s3_bucket_ckpt, s3_object_preprocessing, PROCESSING_CHECKPOINT)
        for i in range(n_ensemble):
            s3_client.download_file(s3_bucket_ckpt, s3_object_classifier[i], CLASSIFY_CHECKPOINT[i])

        s3_client.download_file(s3_bucket_ckpt, s3_sam_mask_generator, SAM_CHECKPOINT)
        s3_client.download_file(s3_bucket_slide, s3_object_key, str(local_SVS_PATH))     
        
        log(inference_row,'RUNNING - Preprocessing')

    # 4. Run model
    trainer.strategy.barrier() ## Sync everything to make sure the data is correctly downloaded
    coord, MI = complete_inference(config,
                                   trainer,
                                   local_SVS_PATH,
                                   PROCESSING_CHECKPOINT,
                                   SAM_CHECKPOINT,
                                   CLASSIFY_CHECKPOINT,
                                   inference_row)
    
    if trainer.is_global_zero:        
        ## Upload the results to S3
        s3_client.upload_file(str(local_result), s3_bucket_slide, s3_result_key)
        print(f"Upload Successful: {s3_result_key}")    
        
        ## Record to the table
        print(coord, MI)
        inference_row.job_progress = 100
        inference_row.job_status   = "SUCCESS"
        inference_row.results      = {"MitoticIndex":str(MI), "CenterX":str(coord[0]), "CenterY":str(coord[1])}

        ## Clean up
        os.remove(local_result)
        os.remove(local_SVS_PATH)
        os.remove(PROCESSING_CHECKPOINT)
        os.remove(SAM_CHECKPOINT)
        for i in range(n_ensemble):
            os.remove(CLASSIFY_CHECKPOINT[i])
        for gpu_id in range(trainer.world_size):
            os.remove(f"{local_SVS_PATH[:-4]}_{gpu_id}.csv")

