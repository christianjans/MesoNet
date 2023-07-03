import argparse
import os
import pickle
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from helpers.image_series import ImageSeriesCreator
from utils import reorder_matrix

REGION_POINTS_WIDTH_MAX = 512
REGION_POINTS_HEIGHT_MAX = 512

REGION_POINTS_AWAKE1 = {
    # Left hemisphere.
    (240, 300): 22,
    (182, 232): 21,
    (154, 252): 27,
    (184, 288): 25,
    (158, 330): 29,
    (132, 310): 24,
    (190, 360): 39,
    (234, 410): 38,
    (188, 452): 35,
    (120, 452): 34,

    # Right hemisphere.
    (276, 300): 18,
    (346, 232): 19,
    (376, 260): 13,
    (342, 294): 15,
    (380, 334): 11,
    (406, 310): 16,
    (340, 366): 1,
    (278, 410): 2,
    (344, 452): 5,
    (426, 452): 6,
}

REGION_POINTS_AWAKE2 = {
    # Left hemisphere.
    (234, 276): 22,
    (164, 228): 21,
    (136, 304): 27,
    (170, 300): 25,
    (136, 348): 29,
    (92, 332): 24,
    (172, 380): 39,
    (234, 414): 38,
    (156, 490): 35,
    (76, 480): 34,

    # Right hemisphere.
    (282, 274): 18,
    (356, 218): 19,
    (386, 258): 13,
    (348, 296): 15,
    (382, 340): 11,
    (422, 324): 16,
    (358, 372): 1,
    (294, 414): 2,
    (392, 460): 5,
    (454, 460): 6,
}

REGION_NAMES = {
    # Left hemisphere.
    22: "L-M2",
    21: "L-M1",
    27: "L-FL",
    25: "L-HL",
    29: "L-UN",
    24: "L-BC",
    39: "L-A",
    38: "L-RS",
    35: "L-V1",
    34: "L-LM",

    # Right hemisphere.
    18: "R-M2",
    19: "R-M1",
    13: "R-FL",
    15: "R-HL",
    11: "R-UN",
    16: "R-BC",
    1: "R-A",
    2: "R-RS",
    5: "R-V1",
    6: "R-LM",
}


# TODO
MATLAB = {
    18: "rM2",
    19: "rM1",
    2: "rRS",
    5: "rV1",
}


# 0 1 2 3 4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
# 1 2 5 6 11 13 15 16 18 19 21 22 24 25 27 29 34 35 38 39
# PAPER_ORDER = [
#     5,
#     14,
#     6,
#     13,
#     10,
#     11,
#     9,
#     8,
#     12,
#     7,
#     15,
#     4,

# ]


class MasksManager:
    def __init__(self,
                 region_points: Union[str, Dict[Tuple[int, int], int]],
                 image_width: int,
                 image_height: int):
        self.image_width = image_width
        self.image_height = image_height

        assert REGION_POINTS_WIDTH_MAX % self.image_width == 0
        assert REGION_POINTS_HEIGHT_MAX % self.image_height == 0

        # Factor to go from region points to image points.
        self.scale_factor_x = self.image_width / REGION_POINTS_WIDTH_MAX
        self.scale_factor_y = self.image_height / REGION_POINTS_HEIGHT_MAX

        if isinstance(region_points, str):
            with open(region_points, 'rb') as f:
                self.region_points: Dict[Tuple[int, int], int] = pickle.load(f)
        elif isinstance(region_points, dict):
            self.region_points = region_points
        else:
            raise ValueError(f"Invalid region_points parameter.")

        self.n_regions = self._determine_n_regions()
        self.masks = np.zeros((self.n_regions,
                               self.image_height,
                               self.image_width), dtype=np.uint8)

        self._populate_masks()

    def _populate_masks(self):
        for point, region in self.region_points.items():
            x_resized, y_resized = self._resize_point(point)
            self.masks[region][y_resized][x_resized] = 1

    def _resize_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        x, y = point
        return int(x * self.scale_factor_x), int(y * self.scale_factor_y)

    def _determine_n_regions(self) -> int:
        return max(self.region_points.values()) + 1


def fft(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    masks_manager = MasksManager(args.region_points_file,
                                 args.image_width,
                                 args.image_height)
    image_series = ImageSeriesCreator.create(args.image_file,
                                             args.image_width,
                                             args.image_height,
                                             args.n_frames)

    fft_values = [[] for _ in range(masks_manager.n_regions)]

    for i, image in enumerate(image_series):
        masked_image = image * masks_manager.masks
        masked_image_average = masked_image.mean(axis=(1, 2))

        for j in range(len(fft_values)):
            fft_values[j].append(masked_image_average[j])

    for i, values in enumerate(fft_values):
        fft_values_array = np.array(values)
        fft = np.fft.fft(fft_values_array)
        power = fft * np.conjugate(fft)  # Compute power spectrum.
        plt.plot(power)
        plt.savefig(os.path.join(args.save_dir, f"fft_{i}.png"))
        plt.clf()


def activity_complements(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    masks_manager = MasksManager(args.region_points_file,
                                 args.image_width,
                                 args.image_height)
    image_series = ImageSeriesCreator.create(args.image_file,
                                             args.image_width,
                                             args.image_height,
                                             args.n_frames)

    data = np.zeros((len(masks_manager.masks), args.n_frames))

    # Record the time series activity data for each region that has a
    # complement.
    for i, image in enumerate(image_series):
        masked_image = image * masks_manager.masks
        numerator = masked_image.sum(axis=(1, 2))
        denominator = masks_manager.masks.sum(axis=(1, 2))
        masked_sums = np.divide(numerator,
                                denominator,
                                out=np.zeros((len(masks_manager.masks),)),
                                where=(denominator != 0))
        # masked_sums = masked_image.sum(axis=(1, 2)) / masks_manager.masks.sum(axis=(1, 2))

        for label in range(len(masks_manager.masks) // 2):
            complement_label = len(masks_manager.masks) - label - 1
            data[label][i] = masked_sums[label]
            data[complement_label][i] = masked_sums[complement_label]

    all_correlations = np.corrcoef(data)
    all_correlations_masked = all_correlations * np.tri(len(masks_manager.masks)) * (1 - np.eye(len(masks_manager.masks)))

    # Plot the complement regions.
    for i in range(len(masks_manager.masks) // 2):
        label = i
        complement_label = len(masks_manager.masks) - label - 1

        # Obtain the correlation (r) value of the two activity plots.
        correlation = all_correlations[label][complement_label]
        if np.isnan(correlation):
            continue
        plot_filename = f"complement_{label}-{complement_label}_r{correlation:.3f}.png"
        print(f"complement correlation: {label}-{complement_label} {correlation}")

        plt.title(f"{label} - {complement_label} activity")
        plt.plot(data[label], label=label)
        plt.plot(data[complement_label], label=complement_label)
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, plot_filename))
        plt.clf()

    # Determine other regions that may be correlated.

    # For awake1 regions.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.90,
    #                                    all_correlations_masked < 1.0))

    # For awake1 regions preprocessed.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.70,
    #                                    all_correlations_masked < 1.0))

    # For awake1 paper.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.70,
    #                                    all_correlations_masked < 1.0))

    # For awake1 paper preprocessed.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.50,
    #                                    all_correlations_masked < 1.0))

    # For awake2 regions.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.999,
    #                                    all_correlations_masked < 1.0))

    # For awake2 regions preprocessed.
    r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.80,
                                       all_correlations_masked < 1.0))

    # For awake2 paper.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.95,
    #                                    all_correlations_masked < 1.0))

    # For awake2 paper preprocessed.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.50,
    #                                    all_correlations_masked < 1.0))

    for r1, r2 in zip(r1s, r2s):
        correlation = all_correlations_masked[r1][r2]
        if np.isnan(correlation):
            continue
        print(f"correlation correlation: {r1}-{r2} {correlation}")
        plot_filename = f"correlation_{r1}-{r2}_r{correlation:.3f}.png"
        plt.title(f"{r1} - {r2} activity")
        plt.plot(data[r1], label=r1)
        plt.plot(data[r2], label=r2)
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, plot_filename))
        plt.clf()

    _plot_correlation_matrix(all_correlations,
                             masks_manager.region_points,
                             args.save_dir)


def activity(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    masks_manager = MasksManager(args.region_points_file,
                                 args.image_width,
                                 args.image_height)
    image_series = ImageSeriesCreator.create(args.image_file,
                                             args.image_width,
                                             args.image_height,
                                             args.n_frames)

    plt.imshow(masks_manager.masks[0])
    plt.show()

    data = [[] for _ in range(len(masks_manager.masks))]

    for _, image in enumerate(image_series):
        masked_image = image * masks_manager.masks
        masked_sums = masked_image.sum(axis=(1, 2)) / masks_manager.masks.sum(axis=(1, 2))

        for j in range(len(masks_manager.masks)):
            data[j].append(masked_sums[j])

    for _, values in enumerate(data):
        plt.plot(values)
    plt.savefig(os.path.join(args.save_dir, f"activity.png"))
    plt.clf()


def _plot_correlation_matrix(
    correlation_matrix: np.array,
    region_points: Dict[Tuple[int, int], int],
    save_dir: str,
) -> np.array:
    # Obtain all possible regions in sorted order.
    sorted_regions = sorted(set(region_points.values()))

    # Initialize an array to hold the correlation matrix without invalid values.
    full_correlation_matrix = np.zeros((len(sorted_regions),
                                        len(sorted_regions)))

    # Fill the full correlation matrix.
    for row_index, row_region in enumerate(sorted_regions):
        for column_index, column_region in enumerate(sorted_regions):
            full_correlation_matrix[row_index][column_index] = \
                correlation_matrix[row_region][column_region]

    try:
        matrix_labels = np.array([f"{region} ({REGION_NAMES[region]})"
                                for region in sorted_regions])
    except KeyError:
        matrix_labels = np.array([f"{region}" for region in sorted_regions])

    # Save the full correlation matrix.
    plt.matshow(full_correlation_matrix)
    plt.rcParams.update({"font.size": 6})
    plt.xticks(range(len(sorted_regions)), labels=matrix_labels, rotation=45)
    plt.yticks(range(len(sorted_regions)), labels=matrix_labels)
    plt.tick_params(axis="x", labelbottom=True)
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "correlation.png"), dpi=200)
    plt.clf()

    # Save the full correlation matrix with upper right masked.
    masked_correlation_matrix = full_correlation_matrix * np.tri(len(sorted_regions)) * (1 - np.eye(len(sorted_regions)))
    plt.matshow(masked_correlation_matrix)
    plt.rcParams.update({"font.size": 6})
    plt.xticks(range(len(sorted_regions)), labels=matrix_labels, rotation=45)
    plt.yticks(range(len(sorted_regions)), labels=matrix_labels)
    plt.tick_params(axis="x", labelbottom=True)
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "correlation_masked.png"), dpi=200)
    plt.clf()

    # Save the full correlation matrix with higher values near the diagonal.
    reordered_matrix, new_order = reorder_matrix(full_correlation_matrix)
    plt.matshow(reordered_matrix)
    plt.rcParams.update({"font.size": 6})
    plt.xticks(range(len(sorted_regions)), labels=matrix_labels[new_order], rotation=45)
    plt.yticks(range(len(sorted_regions)), labels=matrix_labels[new_order])
    plt.tick_params(axis="x", labelbottom=True)
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "correlation_reordered.png"), dpi=200)
    plt.clf()

    # TODO: Save the full correlation matrix with order same as the paper.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-points-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--n-frames", type=int, default=2000)
    args = parser.parse_args()

    # fft(args)
    activity_complements(args)
    # activity(args)
