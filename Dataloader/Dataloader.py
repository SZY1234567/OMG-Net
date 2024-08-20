from monai.data.wsi_reader import WSIReader
import numpy as np
import torch

# tile-wise dataloader.

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, tile_dataset, config=None, target_transform=None, transform=None, ):

        super().__init__()
        self.mag = config['OVERALL'].get('Vis', -1)
        self.patch_size = config['OVERALL']['Patch_Size']
        self.target_transform = target_transform
        self.tile_dataset = tile_dataset  # either a dataframe, or a list of dataframes.
        self.transform = transform
        self.wsi_reader = WSIReader(backend=config['OVERALL']['WSIReader'])
        self.wsi_object_dict = {}

    def get_wsi_object(self, image_path):
        if image_path not in self.wsi_object_dict:
            self.wsi_object_dict[image_path] = self.wsi_reader.read(image_path)
        return self.wsi_object_dict[image_path]                

    def __len__(self):
        return int(self.tile_dataset.shape[0])

    def __getitem__(self, id):  # load patches of size [C, W, H]

        svs_path = self.tile_dataset['SVS_PATH'].iloc[id]
        wsi_obj  = self.get_wsi_object(svs_path)

        level = 0  # processing done at highest zoom.
        x_start = int(self.tile_dataset["coords_x"].iloc[id])
        y_start = int(self.tile_dataset["coords_y"].iloc[id])
        try:
            patches, _ = self.wsi_reader.get_data(wsi=wsi_obj, location=(y_start, x_start), size=self.patch_size,
                                                  level=level)
        except:
            raise ValueError(
                f"Could not read {svs_path}: location={(y_start, x_start)}, image size = {wsi_obj.resolutions['level_dimensions'][level]}.")
        patches = np.swapaxes(patches, 0, 2)

        if self.transform:
            patches = self.transform(patches)
        return patches, id


