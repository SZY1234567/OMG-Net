import os
import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import torch
import nrrd
from PIL import Image

def get_bbox_from_mask(mask):
    pos = np.where(mask != 0)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax, ymax]

def downsampling(img, msk, downscale_factor = 1):
    image_shape = img.shape[:2]
    downsampled_image = img[::downscale_factor, ::downscale_factor, :]
    downsampled_image = Image.fromarray(downsampled_image)
    downsampled_image = np.array(downsampled_image)

    downsampled_mask = msk[::downscale_factor, ::downscale_factor,]
    downsampled_mask = Image.fromarray(downsampled_mask)
    downsampled_mask = np.array(downsampled_mask)
    
    return downsampled_image, downsampled_mask

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, config, df, normalization=None, augmentation=None, inference=False):
        super().__init__()
        self.config = config
        self.df = df
        self.Input_Size = self.config['DATA']['Input_Size']
        self.Patch_Size = self.config['DATA']['Patch_Size']
        self.normalization = normalization
        self.augmentation = augmentation
        self.inference = inference
        self.nrrd_path = self.config['DATA']['nrrd_path']
        self.custom_field_map = {
                'SVS_ID': 'string',
                'top_left': 'int list',
                'center': 'int list',
                'dim': 'int list',
                'vis_level': 'int',
                'diagnosis': 'string',
                'annotation_label': 'string',
                'mask': 'double matrix'}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        img, header = nrrd.read(os.path.join(self.nrrd_path, self.df['nrrd_file'][id]), custom_field_map=self.custom_field_map)
        img = img[:, :, :3].astype(np.uint8)
        msk = np.array(header['mask']>0).astype(np.float32)
        img, msk = downsampling(img, msk, self.config['DATA']['Downscale_Factor'])
        
        self.Input_Size = [int(value / self.config['DATA']['Downscale_Factor']) for value in
                           self.config['DATA']['Input_Size']]
        self.Patch_Size = [int(value / self.config['DATA']['Downscale_Factor']) for value in
                           self.config['DATA']['Patch_Size']]

        if self.Input_Size < self.Patch_Size:
            bbox = get_bbox_from_mask(msk)
            center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]

            center[0] = max(center[0],  self.Input_Size[0]/2)
            center[0] = min(center[0], self.Patch_Size[0] - self.Input_Size[0]/2)
            center[1] = max(center[1], self.Input_Size[1]/2)
            center[1] = min(center[1], self.Patch_Size[1] - self.Input_Size[1]/2)

            img = img[int(center[0] - self.Input_Size[0]/2): int(center[0] + self.Input_Size[0]/2),
                  int(center[1] - self.Input_Size[1]/2): int(center[1] + self.Input_Size[1]/2),:]
            msk = msk[int(center[0] - self.Input_Size[0]/2): int(center[0] + self.Input_Size[0]/2),
                  int(center[1] - self.Input_Size[1]/2): int(center[1] + self.Input_Size[1]/2)]

        if self.augmentation is not None:
            transformed = self.augmentation(image=img, mask=msk)
            img = transformed["image"]
            msk = transformed["mask"]

        if self.normalization is not None:
            img = self.normalization(img)
        else:
            img = torch.as_tensor(img, dtype=torch.float32).permute(2,0,1)

        data = {}
        data['img'] = img
        data['msk'] = torch.as_tensor(msk, dtype=torch.float32).unsqueeze(dim=0)
        del img
        del msk

        if self.inference:
            return data
        else:
            return data, self.df['class'][id]

class DataGeneratorAllCell(torch.utils.data.Dataset):
    def __init__(self, config, df, normalization=None, augmentation=None, inference=False):
        super().__init__()
        self.config = config
        self.df = df
        self.masks_path = self.config['DATA']['masks_path']
        self.normalization = normalization
        self.augmentation = augmentation
        self.inference = inference
        self.nrrd_path = self.config['DATA']['nrrd_path']
        self.custom_field_map = {
                'SVS_ID': 'string',
                'top_left': 'int list',
                'center': 'int list',
                'dim': 'int list',
                'vis_level': 'int',
                'diagnosis': 'string',
                'annotation_label': 'string',
                'mask': 'double matrix'}
        self.Input_Size = self.config['DATA']['Input_Size']
        self.Patch_Size = self.config['DATA']['Patch_Size']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        img, header = nrrd.read(os.path.join(self.nrrd_path, self.df['nrrd_file'][id]), custom_field_map=self.custom_field_map)
        img = img[:, :, :3].astype(np.uint8)
        masks_file = np.load(os.path.join(self.masks_path, self.df['masks_file'][id]), allow_pickle=True)
        msk = np.array(masks_file == self.df['mask_id'][id], dtype='float32')
        img, msk = downsampling(img, msk, self.config['DATA']['Downscale_Factor'])

        if self.Input_Size < self.Patch_Size:
            bbox = get_bbox_from_mask(msk)
            center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
            center[0] = max(center[0],  self.Input_Size[0]/2)
            center[0] = min(center[0], self.Patch_Size[0] - self.Input_Size[0]/2)
            center[1] = max(center[1], self.Input_Size[1]/2)
            center[1] = min(center[1], self.Patch_Size[1] - self.Input_Size[1]/2)

            img = img[int(center[0] - self.Input_Size[0]/2): int(center[0] + self.Input_Size[0]/2),
                  int(center[1] - self.Input_Size[1]/2): int(center[1] + self.Input_Size[1]/2),:]
            msk = msk[int(center[0] - self.Input_Size[0]/2): int(center[0] + self.Input_Size[0]/2),
                  int(center[1] - self.Input_Size[1]/2): int(center[1] + self.Input_Size[1]/2)]

        if self.augmentation is not None:
            transformed = self.augmentation(image=img, mask=msk)
            img = transformed["image"]
            msk = transformed["mask"]

        if self.normalization is not None:
            img = self.normalization(img)
        else:
            img = torch.as_tensor(img, dtype=torch.float32).permute(2,0,1)

        data = {}
        data['img'] = img
        data['msk'] = torch.as_tensor(msk, dtype=torch.float32).unsqueeze(dim=0)
        del img
        del msk

        if self.inference:
            return data
        else:
            return data, self.df['class'][id]


class DataModule(LightningDataModule):
    def __init__(self,
                 df_train, df_val, df_test, config,
                 train_normalization=None, val_normalization=None, augmentation=None,
                 **kwargs):

        super().__init__()
        self.config = config
        self.batch_size = self.config['BASEMODEL']['Batch_Size']
        self.num_of_worker = self.config['BASEMODEL']['Num_of_Worker']
        if self.config['BASEMODEL']["Training_Stratgy"] == "AllCells":
            self.train_data = DataGeneratorAllCell(self.config, df_train,
                                                   normalization=train_normalization, augmentation=augmentation,**kwargs)
            self.val_data = DataGeneratorAllCell(self.config, df_val,
                                                 normalization=val_normalization, augmentation=None, **kwargs)
            self.test_data = DataGeneratorAllCell(self.config, df_test,
                                                  normalization=val_normalization, augmentation=None, **kwargs)
        elif self.config['BASEMODEL']["Training_Stratgy"] == "SelectedCells":
            self.train_data = DataGenerator(self.config, df_train,  normalization=train_normalization, augmentation=augmentation, **kwargs)
            self.val_data = DataGenerator(self.config, df_val,  normalization=val_normalization, augmentation=None, **kwargs)
            self.test_data = DataGenerator(self.config, df_test, normalization=val_normalization, augmentation=None, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_of_worker, pin_memory=False,)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_of_worker, pin_memory=False,)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=self.num_of_worker, pin_memory=False,)

