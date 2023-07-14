import argparse
import pickle
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

BREGMA_POINT_X_INDEX = 13
BREGMA_POINT_Y_INDEX = 14
BREGMA_ROW_INDEX = 3

# Add more matplotlib color maps if needed.
COLOR_MAPS = ["viridis", "plasma", "inferno"]


def calculate_center_of_mass(
    region_points: Dict[Tuple[int, int], int]
) -> np.array:
    n_regions = max(region_points.values()) + 1

    center_of_masses = [None for _ in range(n_regions)]
    center_of_masses = np.zeros((n_regions, 2))

    for i in range(n_regions):
        points_in_region = np.array([key for key, value
                                     in region_points.items()
                                     if value == i])
        if points_in_region.size > 0:
            center_of_mass = np.mean(points_in_region, axis=0)
            center_of_masses[i, :] = center_of_mass
        else:
            center_of_masses[i, :] = (np.nan, np.nan)

    return center_of_masses


def main(args):
    landmark_files = args.landmark_files
    region_points_files = args.region_points_files
    image_files = args.image_files

    assert len(landmark_files) == len(region_points_files)

    distance_data = []

    for landmark_file, region_points_file in zip(landmark_files, region_points_files):
        landmarks = np.genfromtxt(landmark_file, delimiter=",")

        bregma_point = np.array([
            landmarks[BREGMA_ROW_INDEX][BREGMA_POINT_X_INDEX],
            landmarks[BREGMA_ROW_INDEX][BREGMA_POINT_Y_INDEX],
        ])

        print(f"bregma point: {bregma_point}")

        with open(region_points_file, "rb") as f:
            region_points = pickle.load(f)

        center_of_masses = calculate_center_of_mass(region_points)

        distances = bregma_point - center_of_masses
        distances = np.multiply(distances, distances)
        distances = np.sum(distances, axis=-1)
        distances = np.sqrt(distances)

        distanes_table = np.c_[np.array([i for i in range(len(distances))]),
                               distances]
        headers = ["region", "distance from bregma"]
        table = tabulate(distanes_table, headers, tablefmt="fancy_grid")
        print(table)

        distance_data.append(distances)

    distance_data = np.vstack(distance_data)
    print(distance_data)
    # print(distance_data.shape)
    aggregated_data = np.c_[
        np.array([i for i in range(distance_data.shape[1])]),
        np.abs(np.mean(np.diff(distance_data, axis=0), axis=0)),
        np.mean(distance_data, axis=0),
        np.std(distance_data, axis=0),
    ]
    headers = ["region", "absolute difference", "average distance from bregma", "std"]
    table = tabulate(aggregated_data, headers, tablefmt="fancy_grid")
    print(table)

    if image_files:
        assert len(image_files) == len(landmark_files)
        assert len(image_files) <= len(COLOR_MAPS)

        n_segmented_images = len(image_files)

        for i, image_file in enumerate(image_files):
            landmarks = np.genfromtxt(landmark_files[i], delimiter=",")
            bregma_point = np.array([
                landmarks[BREGMA_ROW_INDEX][BREGMA_POINT_X_INDEX],
                landmarks[BREGMA_ROW_INDEX][BREGMA_POINT_Y_INDEX],
            ])

            image = cv2.imread(image_file, cv2.IMREAD_COLOR)

            height, width, _ = image.shape
            extent = (-bregma_point[0], width - bregma_point[0],
                      height - bregma_point[1], -bregma_point[1])

            plt.imshow(image,
                       alpha=(1 / n_segmented_images),
                       extent=extent,
                       cmap=COLOR_MAPS[i])
        plt.show()


if __name__ == "__main__":
    """
    python mesonet/bregma_distance.py \
    --landmark-files /Users/christian/Documents/summer2023/MesoNet/mesonet_outputs/awake1_atlas_brain/dlc_output/tmp_videoDLC_resnet50_atlasAug3shuffle1_1030000.csv \
                    /Users/christian/Documents/summer2023/MesoNet/mesonet_outputs/awake2_atlas_brain/dlc_output/tmp_videoDLC_resnet50_atlasAug3shuffle1_1030000.csv \
    --region-points-files /Users/christian/Documents/summer2023/MesoNet/mesonet_outputs/awake1_atlas_brain/dlc_output/region_points.pkl \
                         /Users/christian/Documents/summer2023/MesoNet/mesonet_outputs/awake2_atlas_brain/dlc_output/region_points.pkl \
    --image-file /Users/christian/Documents/summer2023/MesoNet/mesonet_outputs/awake1_atlas_brain/output_overlay/0_mask_segmented.png \
                 /Users/christian/Documents/summer2023/MesoNet/mesonet_outputs/awake2_atlas_brain/output_overlay/0_mask_segmented.png
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark-files", nargs="+", type=str, required=True)
    parser.add_argument("--region-points-files", nargs="+", type=str, required=True)
    parser.add_argument("--image-files", nargs="+", type=str)
    args = parser.parse_args()

    main(args)
