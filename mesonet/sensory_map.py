import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from helpers.image_series import ImageSeriesCreator
from activity_analyzer import MasksManager

MESOSCALE_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_eye-r/imMean.mat"
REGION_POINTS_FILE = "/Users/christian/Documents/summer2023/MesoNet/mesonet_outputs/isoflurane1_mouse6_eye-r_atlas_brain/dlc_output/region_points.pkl"
SAVE_DIR = "/Users/christian/Documents/summer2023/MesoNet/data/isoflurane1_mouse6_eye-r"
EVENT_FRAME = 29  # NOTE: This is the frame number, not the index.
FPS = 30.0
SCOPE = 1.5

# MESOSCALE_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_fl-r/imMean.mat"
# REGION_POINTS_FILE = "/Users/christian/Documents/summer2023/MesoNet/mesonet_outputs/isoflurane1_mouse6_fl-r_atlas_brain/dlc_output/region_points.pkl"
# SAVE_DIR = "/Users/christian/Documents/summer2023/MesoNet/data/isoflurane1_mouse6_fl-r"
# EVENT_FRAME = 29  # NOTE: This is the frame number, not the index.
# FPS = 30.0
# SCOPE = 1.5

# MESOSCALE_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_hl-r/imMean.mat"
# REGION_POINTS_FILE = "/Users/christian/Documents/summer2023/MesoNet/mesonet_outputs/isoflurane1_mouse6_hl-r_atlas_brain/dlc_output/region_points.pkl"
# SAVE_DIR = "/Users/christian/Documents/summer2023/MesoNet/data/isoflurane1_mouse6_hl-r"
# EVENT_FRAME = 29  # NOTE: This is the frame number, not the index.
# FPS = 30.0
# SCOPE = 1.5

# MESOSCALE_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_whisker-l/imMean.mat"
# REGION_POINTS_FILE = "/Users/christian/Documents/summer2023/MesoNet/mesonet_outputs/isoflurane1_mouse6_whisker-l_atlas_brain/dlc_output/region_points.pkl"
# SAVE_DIR = "/Users/christian/Documents/summer2023/MesoNet/data/isoflurane1_mouse6_whisker-l"
# EVENT_FRAME = 29  # NOTE: This is the frame number, not the index.
# FPS = 30.0
# SCOPE = 1.5


# MESOSCALE_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_eye-r/imMean.mat"
# REGION_POINTS_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_eye-r/Eye_R.mat"
# SAVE_DIR = "/Users/christian/Documents/summer2023/MesoNet/data/isoflurane1_mouse6_eye-r"
# EVENT_FRAME = 29  # NOTE: This is the frame number, not the index.
# FPS = 30.0
# SCOPE = 1.5

# MESOSCALE_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_fl-r/imMean.mat"
# REGION_POINTS_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_fl-r/HL_R.mat"
# SAVE_DIR = "/Users/christian/Documents/summer2023/MesoNet/data/isoflurane1_mouse6_fl-r"
# EVENT_FRAME = 29  # NOTE: This is the frame number, not the index.
# FPS = 30.0
# SCOPE = 1.5

# MESOSCALE_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_hl-r/imMean.mat"
# REGION_POINTS_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_hl-r/HL_R.mat"
# SAVE_DIR = "/Users/christian/Documents/summer2023/MesoNet/data/isoflurane1_mouse6_hl-r"
# EVENT_FRAME = 29  # NOTE: This is the frame number, not the index.
# FPS = 30.0
# SCOPE = 1.5

# MESOSCALE_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_whisker-l/imMean.mat"
# REGION_POINTS_FILE = "/Users/christian/Documents/summer2023/matlab/my_data/isoflurane1_mouse6_whisker-l/HL_R.mat"
# SAVE_DIR = "/Users/christian/Documents/summer2023/MesoNet/data/isoflurane1_mouse6_whisker-l"
# EVENT_FRAME = 29  # NOTE: This is the frame number, not the index.
# FPS = 30.0
# SCOPE = 1.5


def coms_from_region_points(
    region_points: Dict[Tuple[int, int], int]
) -> np.ndarray:
    n_regions = max(region_points.values()) + 1
    coms = np.zeros((n_regions, 2))
    points_list = [[] for _ in range(n_regions)]

    for point, region in region_points.items():
        points_list[region].append(point)

    for i, points in enumerate(points_list):
        points_array = np.array(points)
        coms[i, :] = np.average(points_array, axis=0)

    return coms


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    image_series = ImageSeriesCreator.create(
            MESOSCALE_FILE, 256, 256, "all", property="imMean",
            transpose_axes=(2, 0, 1))
    masks_manager = MasksManager(REGION_POINTS_FILE, 256, 256)

    start_frame_index = EVENT_FRAME  # Get the first frame after the event.
    end_frame_index = int(FPS * SCOPE)
    event_array = image_series.image_array[start_frame_index:end_frame_index]

    event_array_max = np.max(event_array, axis=0)
    max_y, max_x = np.unravel_index(np.argmax(event_array_max),
                                    event_array_max.shape)

    roi_mask = masks_manager.masks.copy()
    roi_mask = np.logical_or.reduce(roi_mask, axis=0)
    roi_mask = np.ma.masked_where(roi_mask == 0, roi_mask)

    roi_maxes = event_array_max * masks_manager.masks
    roi_maxes = [np.unravel_index(np.argmax(roi_max), roi_max.shape)
                 for roi_max in roi_maxes]
    roi_maxes = np.array(roi_maxes)

    coms = coms_from_region_points(masks_manager.region_points)

    plt.imshow(event_array_max)
    plt.colorbar()
    plt.imshow(roi_mask, alpha=0.3, cmap="autumn")
    plt.plot(coms[:, 0] / 2, coms[:, 1] / 2, "og", markersize=4)
    plt.plot(roi_maxes[:, 1], roi_maxes[:, 0], "ob", markersize=4)
    plt.plot(max_x, max_y, "xr", markersize=4)
    plt.legend(("COM of ROI", "ROI max", "Sensory peak"))
    plt.savefig(os.path.join(SAVE_DIR, "sensory_map.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
