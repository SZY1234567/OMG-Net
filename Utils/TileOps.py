from monai.data.wsi_reader import WSIReader
import numpy as np
import os
import pandas as pd
from scipy.signal import convolve2d
from sklearn import preprocessing
import time
from tqdm import tqdm


def get_nonbackground_tiles(SVS_PATH, config, pixel_intensity_for_background=240, background_fraction_threshold=0.7):
    # pixel_intensity_for_background : pixel grayscale (0 to 255) value to be considered as background. Value is then corrected using the WSI's perimeter.
    # background_fraction_threshold  : in a tile, fraction of pixels that must be background for that tile to be classified as background.

    # (1) read the image in the lowest visibility
    wsi_reader = WSIReader(backend='CuCIM')
    wsi_obj = wsi_reader.read(SVS_PATH)
    num_levels = wsi_obj.resolutions['level_count']
    low_zoom_level = int(num_levels) - 1
    w, h = wsi_obj.resolutions['level_dimensions'][-1]
    low_zoom_downsample = int(wsi_obj.resolutions['level_downsamples'][-1])
    image_low_zoom, _ = wsi_reader.get_data(wsi=wsi_obj, location=(0, 0), size=(h, w), level=low_zoom_level)
    image_low_zoom = np.swapaxes(image_low_zoom, 0, 2)
    image_low_zoom = image_low_zoom[:, :, 0] * 0.2989 + image_low_zoom[:, :, 1] * 0.5870 + image_low_zoom[:, :,
                                                                                           2] * 0.1140  # grayscale

    # (2) refine background threshold estimate and mask
    # Hypothesis: the perimeter of the WSI is mostly background, and not tissue everywhere. Estimate background from that.
    border_width = 30
    perimeter = np.concatenate((image_low_zoom[0:border_width, :].flatten(),
                                image_low_zoom[-border_width:, :].flatten(),
                                image_low_zoom[:, 0:border_width].flatten(),
                                image_low_zoom[:, -border_width:].flatten()))
    # Set to minimum between existing pixel_intensity_for_background and new value for robustness, in case perimeter is mostly tissue.
    pixel_intensity_for_background = np.minimum(np.mean(perimeter), pixel_intensity_for_background)

    image_low_zoom = image_low_zoom < pixel_intensity_for_background  # mask non-background tiles

    # (3) remove edges of the image so that the convolution is well behaved
    reduced_patch_size = [int(p / low_zoom_downsample) for p in config['OVERALL']['Patch_Size']]
    Cx = image_low_zoom.shape[0] - (image_low_zoom.shape[0] % reduced_patch_size[0])
    Cy = image_low_zoom.shape[1] - (image_low_zoom.shape[1] % reduced_patch_size[1])
    image_low_zoom = image_low_zoom[0:Cx, 0:Cy]

    # (4) perform the convolution operation with the patch size
    kernel = np.ones(reduced_patch_size, dtype=np.float32)
    sum_px = convolve2d(image_low_zoom, kernel, mode='valid', boundary='fill', fillvalue=0)
    downsampled_image = sum_px[::reduced_patch_size[0], ::reduced_patch_size[1]] / np.prod(reduced_patch_size)

    # downsampled_image is the fraction of pixels in each tile which are identified as NOT background. A tile is classified
    # as NOT background if at least (1-background_fraction_threshold)% of its pixels were classified as NOT background.
    non_background_mask = downsampled_image >= (1 - background_fraction_threshold)

    # (5) generate the coordinates that correspond to non-background, in the high zoom space!
    edges_x = np.arange(0, image_low_zoom.shape[0], reduced_patch_size[0])
    edges_y = np.arange(0, image_low_zoom.shape[1], reduced_patch_size[1])
    EY, EX = np.meshgrid(edges_y, edges_x)
    EX_high_zoom = EX[non_background_mask] * low_zoom_downsample
    EY_high_zoom = EY[non_background_mask] * low_zoom_downsample
    corners_without_background = np.column_stack((EX_high_zoom, EY_high_zoom))

    return corners_without_background


def getTilesFromAnnotations(config=None, SVS_dataset=None, QA_contours=False):
    patch_size = config['Patch_Size']

    df = pd.DataFrame()
    # Process contour-wise
    for idx, row in tqdm(SVS_dataset.iterrows(), total=len(SVS_dataset), desc="Extracting tiles from annotations..."):
        cur_dataset = contours_processing(row, patch_size)
        cur_dataset['id_external'] = row['id_external']
        cur_dataset['SVS_PATH'] = row['SVS_PATH']
        df = pd.concat([df, cur_dataset], ignore_index=True)

    df[['coords_x', 'coords_y']] = df[['coords_x', 'coords_y']].astype('int')

    # For display/QA purposes; reorganise as WSI-wise and overlay contour
    if QA_contours:
        # df_final = pd.DataFrame()
        for name, group in df.groupby('id_external'):
            group = group.drop_duplicates(subset=['coords_x', 'coords_y'], keep='last')
            # df_final = pd.concat([df_final, group],ignore_index=True)
            Create_Contours_Overlay_QA(group, config)

    return df


def contours_processing(row, patch_size):
    # Process one contour in a WSI.
    coords = row['Points']

    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)

    # To make sure we do not end up with overlapping contours at the end, round xmin, xmax, ymin,
    # ymax to the nearest multiple of patch_size.
    xmin = np.floor(xmin / patch_size[0]).astype(np.int) * patch_size[0]
    ymin = np.floor(ymin / patch_size[1]).astype(np.int) * patch_size[1]
    xmax = np.ceil(xmax / patch_size[0]).astype(np.int) * patch_size[0]
    ymax = np.ceil(ymax / patch_size[1]).astype(np.int) * patch_size[1]

    # Find the meshgrid of points included into the contour
    xx, yy = np.meshgrid(np.arange(xmin, xmax, patch_size[0]), np.arange(ymin, ymax, patch_size[0]))
    points = np.vstack((xx.flatten(), yy.flatten())).T
    poly_path = Path(coords)
    inside = poly_path.contains_points(points)
    in_pts = points[inside]

    return pd.DataFrame({'coords_x': [p[0] for p in in_pts],
                         'coords_y': [p[1] for p in in_pts],
                         'tissue_type': row['ROIName']})


def Create_Contours_Overlay_QA(df_export, config, QA_folder=None):
    # Create export folder
    if QA_folder is None:
        QA_folder = os.path.join(os.getcwd(), 'QA_patches')
    os.makedirs(QA_folder, exist_ok=True)

    # Convert labels to numerical values
    le = preprocessing.LabelEncoder()
    numerical_labels = le.fit_transform(df_export['tissue_type'])

    # Read WSI at lowest zoom
    wsi_reader = WSIReader(backend='CuCIM')
    wsi_obj = wsi_reader.read(df_export['SVS_PATH'].iloc[0])
    num_levels = wsi_obj.resolutions['level_count']
    W, H = wsi_obj.resolutions['level_dimensions'][-1]
    downsample_factor = int(wsi_obj.resolutions['level_downsamples'][-1])
    image_low_zoom, _ = wsi_reader.get_data(wsi=wsi_obj, location=(0, 0), size=(H, W), level=int(num_levels) - 1)
    image_low_zoom = image_low_zoom.transpose(1, 2, 0)  # H, W, C

    coords = np.array(df_export[["coords_x", "coords_y"]]) / downsample_factor  # display the corner of patches

    # ------------ create figure

    # Dynamic figure size with a bit more width
    cmap = plt.cm.get_cmap('Set1')
    norm = mpl.colors.Normalize(vmin=numerical_labels.min(), vmax=numerical_labels.max())
    aspect_ratio = image_low_zoom.shape[1] / image_low_zoom.shape[0]
    fig_width = 12
    fig_height = fig_width / aspect_ratio
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = gridspec.GridSpec(1, 2, width_ratios=[30, 1])

    # Image subplot
    ax_image = plt.subplot(gs[0])
    ax_image.imshow(image_low_zoom)
    ax_image.set_title("Whole Slide Image with contours overlay", pad=20)
    ax_image.axis('off')

    scatter = ax_image.scatter(coords[:, 0], coords[:, 1], c=numerical_labels, cmap=cmap, norm=norm)

    # Color bar subplot
    ax_cbar = plt.subplot(gs[1])
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('Tissue type')

    # Adjust layout
    plt.tight_layout()

    filename = os.path.join(QA_folder, df_export['id_external'].iloc[0] + '.pdf')
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Overlay with contours exported at {filename}")
