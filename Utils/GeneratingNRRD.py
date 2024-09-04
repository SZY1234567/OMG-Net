import json
import os
import gc
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import openslide
import sqlite3
import matplotlib.pyplot as plt
import nrrd
import torch
from segment_anything import sam_model_registry, SamPredictor

def get_img_msk(
        filename,
        center_x, 
        center_y, 
        label, 
        nrrd_file, 
        image_folder, 
        nrrd_folder, 
        predictor, 
        dim=[256, 256], 
        vis_level=0,
        box_dim=[32, 32],
        ):
    
    slide = openslide.open_slide(os.path.join(image_folder, filename))
    top_left = (int(center_x - dim[0]/2), int(center_y - dim[1]/2))
    center_x = int(center_x - top_left[0])
    center_y = int(center_y - top_left[1])
    img = np.array(slide.read_region(top_left, level=vis_level, size=(dim[0], dim[1])))[:, :, :3]
    bbox = [center_x - int(box_dim[0]/2), center_y - int(box_dim[1]/2),
            center_x + int(box_dim[0]/2), center_y + int(box_dim[1]/2), ]

    header = {
        'filename': filename,
        'top_left': top_left,
        'center': (center_x, center_y),
        'dim': (dim[0], dim[1]),
        'vis_level': vis_level,
        'annotation_label': 1 if label == 'mitotic figure' else 0,
        'mask': np.zeros_like(img)
        }

    predictor.set_image(img)
    masks, _, _ = predictor.predict(
        box=np.array([bbox[0], bbox[1], bbox[2], bbox[3],])[None, :],
        multimask_output=False,
    )

    header['mask'] = np.array(masks[0]).astype('float32')
    nrrd.write(os.path.join(nrrd_folder, '{}.nrrd'.format(nrrd_file)), img, header, custom_field_map=custom_field_map)

def save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim=[256, 256], vis_level=0, box_dim=[32, 32]):
    for i in range(len(df)):
        get_img_msk(df['filename'][i], df.coordinateX[i], df.coordinateY[i], 
                    df['annotation_label'][i], df['nrrd_file'][i], 
                    image_folder, nrrd_folder, predictor, 
                    dim, vis_level, box_dim)

def table2df(cursor, table_name):
    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    return df

#Configure Paths
nrrd_folder        = "/path/to/nrrd"                #Path to save the NRRD files
midog_folder       = "/path/to/MIDOG_PLUS/"         #Path to MIDOG_PLUS
cmc_folder         = "/path/to/MITOS_WSI_CMC/"      #Path to MITOS_WSI_CMC
ccmct_folder       = "/path/to/MITOS_WSI_CCMCT/"    #Path to MITOS_WSI_CCMCT
tupac_folder       = "/path/to/TUPAC/"              #Path to TUPAC

#Load SAM mask generator
sam_checkpoint     = "sam_vit_h_4b8939.pth" 
model_type         = "vit_h"
device             = "cuda"
dim                = [256, 256]
vis_level          = 0
box_dim            = [32, 32]
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=torch.device(device))
predictor = SamPredictor(sam)

#Configure NRRD files
custom_field_map   = {
    'SVS_ID': 'string',
    'top_left': 'int list',
    'center': 'int list',
    'dim': 'int list',
    'vis_level': 'int',
    'diagnosis': 'string',
    'annotation_label': 'string',
    'mask': 'double matrix'
    }

#Process the MIDOG dataset-----------------------------------------------------------------------------------------------
print("Processing MIDOG dataset")
annotation_file = os.path.join(midog_folder, "MIDOG++.json")
image_folder    = os.path.join(midog_folder, "images")
image_files     = [fn.split("\\")[-1] for fn in glob.glob(image_folder + "/*.tiff")]
slides          = pd.read_csv(os.path.join(midog_folder, "datasets_xvalidation.csv"), delimiter=";")
dataframe       = os.path.join(midog_folder, "MIDOG.csv")

print("Creating dataframe from annotation")
rows = []
with open(annotation_file) as f:
    data = json.load(f)
    categories = {1: 'mitotic figure', 2: 'hard negative'}
    for row in data["images"]:
        file_name = row["file_name"]
        image_id = row["id"]
        width = row["width"]
        height = row["height"]
        for ann_id, annotation in enumerate([anno for anno in data['annotations'] if anno["image_id"] == image_id]):
            box = annotation["bbox"]
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cat = categories[annotation["category_id"]]
            slide = slides.loc[slides['Slide'] == image_id, 'Tumor':]
            tumour = slide['Tumor'].array[0]
            scanner = slide['Scanner'].array[0]
            origin = slide['Origin'].array[0]
            species = slide['Species'].array[0]
            nrrd_file = 'MIDOG_{}_{}.nrrd'.format(f"{image_id:03d}", ann_id)
            rows.append(
                [file_name, image_id, ann_id, width, height, xmin, ymin, xmax, ymax, cat, tumour, scanner, origin,
                    species, nrrd_file])

df = pd.DataFrame(rows, columns=["filename", "image_id", "ann_id", "width", "height",
                                 "xmin", "ymin", "xmax", "ymax",
                                 "annotation_label", "tumour", "scanner", "origin", "species", "nrrd_file"])
df['coordinateX'] = (df['xmin'] + df['xmax']) / 2
df['coordinateY'] = (df['ymin'] + df['ymax']) / 2
df.to_csv(dataframe, index=False)

save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim, vis_level, box_dim)

#Process the MITOS_WSI_CMC/MITOS_WSI_CCMCT dataset-----------------------------------------------------------------------------------------------
print("Processing MITOS_WSI_CMC/MITOS_WSI_CCMCT dataset")
for dataset in ['MITOS_WSI_CMC', 'MITOS_WSI_CCMCT']:
    if dataset == "MITOS_WSI_CCMCT":
        annotation_file = os.path.join(ccmct_folder, "databases", "MITOS_WSI_CCMCT_ODAEL.sqlite")
        image_folder    = os.path.join(ccmct_folder, "WSI")
        dataframe       = os.path.join(ccmct_folder, "MITOS_WSI_CCMCT.csv")
        mitotic_label   = 'mitotic figure'

    elif dataset == "MITOS_WSI_CMC":
        annotation_file = os.path.join(cmc_folder, "databases", "MITOS_WSI_CMC_CODAEL_TR_ROI.sqlite")
        image_folder    = os.path.join(cmc_folder, "WSI")
        dataframe       = os.path.join(cmc_folder, "MITOS_WSI_CMC.csv")
        mitotic_label   = 'Mitotic figure'
    
    print("Creating dataframe from annotation")
    con = sqlite3.connect(annotation_file)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

    df_Annotations = table2df(cur, 'Annotations')
    df_sqlite_sequence = table2df(cur, 'sqlite_sequence')
    df_Annotations_coordinates = table2df(cur, 'Annotations_coordinates')
    df_Annotations_label = table2df(cur, 'Annotations_label')
    df_Classes = table2df(cur, 'Classes')
    df_Log = table2df(cur, 'Log')
    df_Persons = table2df(cur, 'Persons')
    df_Slides = table2df(cur, 'Slides')
    con.close()

    df_Annotations = df_Annotations[df_Annotations['agreedClass'].isin([1, 2])]
    df_Annotations_coordinates = df_Annotations_coordinates[df_Annotations_coordinates['orderIdx'] == 1]
    df_Annotations_coordinates.drop(columns=['slide'], inplace=True)
    df_Annotations = df_Annotations.rename(columns={'uid': 'annoId'})

    df_Slides = df_Slides.rename(columns={'uid': 'slide'})
    df_Classes = df_Classes.rename(columns={'uid': 'agreedClass', 'name': 'annotation_label'})

    df = df_Annotations.merge(df_Slides, on='slide', how='inner')
    df = df.merge(df_Annotations_coordinates, on='annoId', how='inner')
    df = df.merge(df_Classes, on='agreedClass', how='inner')
    df = df.replace(mitotic_label, 'mitotic figure')
    df.reset_index(drop=True, inplace=True)
    df.to_csv(dataframe, index=False)

    save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim, vis_level, box_dim)

#Process the TUPAC16 dataset-----------------------------------------------------------------------------------------------
print("Processing TUPAC16 dataset")
annotation_file = os.path.join(tupac_folder, "databases", "TUPAC_alternativeLabels_augmented_training.sqlite")
image_folder    = os.path.join(tupac_folder, "WSI")
dataframe       = os.path.join(tupac_folder, "TUPAC16.csv")

print("Creating dataframe from annotation")
con = sqlite3.connect(annotation_file)
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

df_Annotations = table2df(cur, 'Annotations')
df_sqlite_sequence = table2df(cur, 'sqlite_sequence')
df_Annotations_coordinates = table2df(cur, 'Annotations_coordinates')
df_Annotations_label = table2df(cur, 'Annotations_label')
df_Classes = table2df(cur, 'Classes')
df_Log = table2df(cur, 'Log')
df_Persons = table2df(cur, 'Persons')
df_Slides = table2df(cur, 'Slides')

print(df_Annotations.agreedClass.value_counts())
print(df_Annotations.uid.unique().shape)
print(df_Annotations_label.annoId.unique().shape)
print(df_Annotations_coordinates.annoId.unique().shape)
con.close()

df_Annotations_coordinates.drop(columns=['slide'], inplace=True)
df_Annotations = df_Annotations.rename(columns={'uid': 'annoId'})
df_Slides = df_Slides.rename(columns={'uid': 'slide'})
df_Classes = df_Classes.rename(columns={'uid': 'agreedClass', 'name': 'annotation_label'})

df = df_Annotations.merge(df_Slides, on='slide', how='inner')
df = df.merge(df_Annotations_coordinates, on='annoId', how='inner')
df = df.merge(df_Classes, on='agreedClass', how='inner')
df.reset_index(drop=True, inplace=True)
df.drop(columns=['guid', 'lastModified', 'deleted', 'type',
                    'description', 'directory', 'uuid', 'exactImageID',
                    'EXACTUSER', 'uid', 'orderIdx', 'coordinateZ', 'color'], inplace=True)
df = df.replace('Mitose', 'mitotic figure')
df.to_csv(dataframe, index=False)

save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim, vis_level, box_dim)



