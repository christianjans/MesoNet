import argparse
import os
import pickle
from typing import Dict, List, Tuple, Union

import cv2
import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy

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
    # Right hemisphere.
    18: "rM2",
    19: "rM1",
    2: "rRS",
    5: "rV1",
    13: "rFL",
    15: "rHL",
    16: "rBC",
    12: "rTR",
    14: "rMO",
    17: "rNO",
    # ??: "rS2",
    9: "rAU",
    # ??: "rTEA",
    # ??: "rUNa",
    # ??: "rUNb",
    # ??: "rCG",
    # ??: "rPTAa",
    # ??: "rPTAb",

    # Left hemisphere.
    22: "lM2",
    21: "lM1",
    38: "lRS",
    35: "lV1",
    27: "lFL",
    25: "lHL",
    24: "lBC",
    28: "lTR",
    26: "lMO",
    23: "lNO",
    # ??: "lS2",
    31: "lAU",
    # ??: "lTEA",
    # ??: "lUNa",
    # ??: "lUNb",
    # ??: "lCG",
    # ??: "lPTAa",
    # ??: "lPTAb",
}
MATLAB_INVERSE = {value: key for key, value in MATLAB.items()}


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


class RegionPointsWrapper:
    def __init__(self, region_points: Dict[Tuple[int, int], int]):
        self._region_points = self._transform_region_points(region_points)

    def _transform_region_points(
        self, region_points: Dict[Tuple[int, int], int]
    ) -> Dict[Tuple[int, int], int]:
        raise NotImplementedError

    @property
    def region_points(self) -> Dict[Tuple[int, int], int]:
        return self._region_points

    def regions(self) -> List[int]:
        raise NotImplementedError

    def labels(self) -> List[str]:
        raise NotImplementedError


class CenterOfMassRegionPoints(RegionPointsWrapper):
    def __init__(self, region_points: Dict[Tuple[int, int], int]):
        super().__init__(region_points)

    def _transform_region_points(
        self, region_points: Dict[Tuple[int, int], int]
    ) -> Dict[Tuple[int, int], int]:
        return super()._transform_region_points()
    
    def regions(self) -> List[int]:
        return super().regions()
    
    def labels(self) -> List[str]:
        return super().labels()


class HalvesRegionPoints(RegionPointsWrapper):
    REGIONS_OF_INTEREST = [22, 38, 18, 2]

    def __init__(self, region_points: Dict[Tuple[int, int], int]):
        super().__init__(region_points)

    def _transform_region_points(
        self, region_points: Dict[Tuple[int, int], int]
    ) -> Dict[Tuple[int, int], int]:
        new_region_points: Dict[Tuple[int, int], int] = {}
        regions_of_interest_points = [
            [
                point for point, region in region_points.items()
                if region == region_of_interest
            ]
            for region_of_interest in HalvesRegionPoints.REGIONS_OF_INTEREST
        ]

        for region, region_of_interest_points in zip(HalvesRegionPoints.REGIONS_OF_INTEREST, regions_of_interest_points):
            min_height = min(region_of_interest_points, key=lambda point: point[1])[1]
            max_height = max(region_of_interest_points, key=lambda point: point[1])[1]
            middle_height = (max_height + min_height) // 2

            y = (max_height + middle_height) // 2
            points = [point for point in region_of_interest_points
                    if point[1] == y]
            left = min(points, key=lambda point: point[0])[0]
            right = max(points, key=lambda point: point[0])[0]
            x = (left + right) // 2
            point = (x, y)
            new_region_points[point] = region

            y = (min_height + middle_height) // 2
            points = [point for point in region_of_interest_points
                    if point[1] == y]
            left = min(points, key=lambda point: point[0])[0]
            right = max(points, key=lambda point: point[0])[0]
            x = (left + right) // 2
            point = (x, y)
            new_region_points[point] = region + 1

        return new_region_points
    
    def regions(self) -> List[int]:
        return super().regions()
    
    def labels(self) -> List[str]:
        return super().labels()


def transform_region_points(
    region_points: Dict[Tuple[int, int], int]
) -> Dict[Tuple[int, int], int]:
    new_region_points: Dict[Tuple[int, int], int] = {}

    regions_of_interest = [22, 38, 18, 2]
    regions_of_interest_points = [
        [
            point for point, region in region_points.items()
            if region == region_of_interest
        ]
        for region_of_interest in regions_of_interest
    ]

    for region, region_of_interest_points in zip(regions_of_interest, regions_of_interest_points):
        min_height = min(region_of_interest_points, key=lambda point: point[1])[1]
        max_height = max(region_of_interest_points, key=lambda point: point[1])[1]
        middle_height = (max_height + min_height) // 2

        y = (max_height + middle_height) // 2
        points = [point for point in region_of_interest_points
                  if point[1] == y]
        left = min(points, key=lambda point: point[0])[0]
        right = max(points, key=lambda point: point[0])[0]
        x = (left + right) // 2
        point = (x, y)
        new_region_points[point] = region

        y = (min_height + middle_height) // 2
        points = [point for point in region_of_interest_points
                  if point[1] == y]
        left = min(points, key=lambda point: point[0])[0]
        right = max(points, key=lambda point: point[0])[0]
        x = (left + right) // 2
        point = (x, y)
        new_region_points[point] = region + 1

    return new_region_points


class MasksManager:
    def __init__(self,
                 region_points: Union[str, Dict[Tuple[int, int], int]],
                 image_width: int,
                 image_height: int,
                 use_center_of_mass: bool = False,
                 square_center_of_mass_points: bool = False):
        self.image_width = image_width
        self.image_height = image_height
        self.use_center_of_mass = use_center_of_mass
        self.square_center_of_mass_points = square_center_of_mass_points

        assert REGION_POINTS_WIDTH_MAX % self.image_width == 0
        assert REGION_POINTS_HEIGHT_MAX % self.image_height == 0

        # Factor to go from region points to image points.
        self.scale_down_factor_x = self.image_width / REGION_POINTS_WIDTH_MAX
        self.scale_down_factor_y = self.image_height / REGION_POINTS_HEIGHT_MAX
        self.scale_up_factor_x = REGION_POINTS_WIDTH_MAX // self.image_width
        self.scale_up_factor_y = REGION_POINTS_HEIGHT_MAX // self.image_height

        if isinstance(region_points, str):
            if region_points.endswith(".pkl"):
                with open(region_points, "rb") as f:
                    self.region_points: Dict[Tuple[int, int], int] = pickle.load(f)
            elif region_points.endswith(".mat"):
                # assert "bilatregionalcorr" in region_points.lower()
                self.region_points = self._region_points_from_mat(region_points)
            else:
                raise ValueError(
                        f"Unrecognized region points file {region_points}")
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
        self.region_points_mask = self.masks.copy()

        # # TODO: Remove
        # self.region_points = transform_region_points(self.region_points)
        # self.new_masks = np.zeros((self.n_regions,
        #                            self.image_height,
        #                            self.image_width), dtype=np.uint8)
        # for point, region in self.region_points.items():
        #     x_resized, y_resized = self._resize_point(point)
        #     self.masks[region][y_resized][x_resized] = 1

        # TODO: Make this cleaner.
        if self.use_center_of_mass:
            self._calculate_center_of_mass(self.square_center_of_mass_points)

    def _resize_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        x, y = point
        return int(x * self.scale_down_factor_x), int(y * self.scale_down_factor_y)

    def _determine_n_regions(self) -> int:
        return max(self.region_points.values()) + 1

    def _calculate_center_of_mass(self, square: bool):
        new_masks = np.zeros_like(self.masks)
        for i in range(len(self.masks)):
            ys, xs = np.where(self.masks[i] == 1)
            if len(xs) > 0 and len(ys) > 0:
                centroid_x = int(np.average(xs))
                centroid_y = int(np.average(ys))
                new_masks[i][centroid_y][centroid_x] = 1
                if square:
                    for column in range(centroid_x - 5, centroid_x + 5):
                        for row in range(centroid_y - 5, centroid_y + 5):
                            new_masks[i][row][column] = 1
        self.masks = new_masks

    def _region_points_from_mat(self, mat_file: str) -> Dict[Tuple[int, int], int]:
        region_points: Dict[Tuple[int, int], int] = {}

        try:
            mat = scipy.io.loadmat(mat_file)
            roi_points = mat["RHR"]
            roi_labels = [label[0] for label in mat["ROIlabels"][0]]
        except NotImplementedError:
            with h5py.File(mat_file) as f:
                # Obtain the ROI points.
                roi_points = f["RHR"][:]
                roi_points = roi_points.T
                roi_points = roi_points.astype(np.int32)

                # Obtain the number of ROI points.
                n_roi_points = int(f["nRHR"][0][0])
                assert len(roi_points) == n_roi_points

                # Obtain the ROI labels.
                roi_labels = []
                for i in range(n_roi_points):
                    label_reference = f["ROIlabels"][i][0]
                    label_object = f[label_reference]
                    roi_labels.append(
                        "".join([chr(j) for j in np.squeeze(label_object[:])])
                    )

        assert len(roi_points) == len(roi_labels)

        for label, (x, y) in zip(roi_labels, roi_points):
            if label not in MATLAB_INVERSE:
                continue

            region_number = MATLAB_INVERSE[label]
            x, y = x - 1, y - 1
            for row in range(y - 5, y + 5 + 1):
                for column in range(x - 5, x + 5 + 1):
                    new_y = self.scale_up_factor_x * row
                    new_x = self.scale_up_factor_x * column
                    region_points[(new_x, new_y)] = region_number

        return region_points


def fft(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    masks_manager = MasksManager(args.region_points_file,
                                 args.image_width,
                                 args.image_height)
    image_series = ImageSeriesCreator.create_cached_image_series(
            args.image_file, args.image_width, args.image_height, args.n_frames)

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
                                 args.image_height,
                                 use_center_of_mass=args.use_com,
                                 square_center_of_mass_points=args.square_com)
    image_series = ImageSeriesCreator.create_cached_image_series(
            args.image_file, args.image_width, args.image_height, args.n_frames)
    # image_series = ImageSeriesCreator.create(args.image_file,
    #                                          args.image_width,
    #                                          args.image_height,
    #                                          args.n_frames,
    #                                          property=args.mat_property,
    #                                          transpose_axes=args.mat_transpose_axes)

    # Save masks on images if a still image is provided.
    if args.still_image_file:
        # Plot the mesoscale image of the brain.
        still_image = cv2.imread(args.still_image_file, cv2.IMREAD_UNCHANGED)
        still_image = cv2.resize(still_image, (args.image_height, args.image_width))
        plt.imshow(still_image)        

        # Plot the predicted ROIs from MesoNet.
        region_points = np.zeros((args.image_height, args.image_width))
        for x, y in masks_manager.region_points:
            new_x = int(x * masks_manager.scale_down_factor_x)
            new_y = int(y * masks_manager.scale_down_factor_y)
            region_points[new_y][new_x] = 1
        transparent_region_points = np.ma.masked_where(region_points == 0, region_points)
        plt.imshow(transparent_region_points, alpha=0.6)

        # Plot the current masks that we use.
        for i, mask in enumerate(masks_manager.masks):
            transparent_mask = np.ma.masked_where(mask == 0, mask)
            plt.imshow(transparent_mask, alpha=0.1, cmap="autumn")

            ys, xs = np.where(mask == 1)
            if len(xs) > 0 and len(ys) > 0:
                x, y = xs[0], ys[0]
                plt.annotate(f"{i}",
                             xy=(x, y),
                             xytext=(x, y),
                             color="#FFFFFF",
                             fontsize=7)
        plt.savefig(os.path.join(args.save_dir, "masks.png"), dpi=200)
    plt.clf()

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

        for label in range(len(masks_manager.masks)):
            data[label][i] = masked_sums[label]

    all_correlations = np.corrcoef(data)
    all_correlations_masked = all_correlations * np.tri(len(masks_manager.masks)) * (1 - np.eye(len(masks_manager.masks)))

    np.save(os.path.join(args.save_dir, "timecourse.npy"), data)

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

    # Plot the highlighted regions.
    for highlight in args.highlights:
        plot_filename = f"activity_{highlight}.png"
        plt.title(f"{highlight} activity")
        plt.plot(data[highlight], label=highlight)
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, plot_filename))
        plt.clf()

    # Determine other regions that may be correlated.

    # For awake1 regions preprocessed.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.70,
    #                                    all_correlations_masked < 1.0))

    # For awake1 paper preprocessed.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.50,
    #                                    all_correlations_masked < 1.0))

    # For awake2 regions preprocessed.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.80,
    #                                    all_correlations_masked < 1.0))

    # For awake2 paper preprocessed.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.50,
    #                                    all_correlations_masked < 1.0))

    # For awake1/2 com preprocessed.
    r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.80,
                                       all_correlations_masked < 1.0))

    # For awake1/2 square com preprocessed.
    # r1s, r2s = np.where(np.logical_and(all_correlations_masked > 0.80,
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
    image_series = ImageSeriesCreator.create_cached_image_series(
            args.image_file, args.image_width, args.image_height, args.n_frames)

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


def seed_pixel_map(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load the region points file.
    with open(args.region_points_file, "rb") as f:
        region_points = pickle.load(f)
    region_points = transform_region_points(region_points)

    # BACKGROUND_IMAGE = "/Users/christian/Documents/summer2023/MesoNet/mesonet_inputs/awake1_data/atlas_brain/0.png"
    BACKGROUND_IMAGE = "/Users/christian/Documents/summer2023/MesoNet/mesonet_inputs/awake2_data/atlas_brain/0.png"
    assert False, "Did you check that the above image path is correct?"
    background_image = cv2.imread(BACKGROUND_IMAGE, cv2.IMREAD_GRAYSCALE)

    image_series = ImageSeriesCreator.create_cached_image_series(
            args.image_file, args.image_width, args.image_height, args.n_frames)

    data = np.zeros((args.n_frames,
                     args.image_height,
                     args.image_width), dtype=np.float64)

    for i, image in enumerate(image_series):
        data[i] = image

    data = np.transpose(np.reshape(data, (args.n_frames, -1)))
    correlation = np.corrcoef(data)

    np.save(os.path.join(args.save_dir, "correlation.npy"), correlation)

    figure, axes = plt.subplots(nrows=1, ncols=len(region_points))
    figure.set_size_inches(2 * len(region_points), 4)
    figure.subplots_adjust(wspace=0.5)
    for i, (x, y) in enumerate(region_points):
        new_x, new_y = x // 4, y // 4
        map = correlation[new_y * 128 + new_x, :]
        map = np.reshape(map, (128, 128))
        dot = patches.Circle((new_x, new_y), 1, edgecolor="black")
        axes[i].imshow(background_image)
        axes[i].imshow(map, alpha=0.7)
        axes[i].add_patch(dot)
        axes[i].set_title((new_x, new_y))
    plt.savefig(os.path.join(args.save_dir, "seed_pixel_map.png"))
    plt.show()


def correlation_matrix_comparison():
    # NOTE: To add more regions, modify the MATLAB dictionary above.

    DATASET = "awake2"  # TODO
    SAVE_DIR = f"./data/corrmatcomp_{DATASET}_test"

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    MESONET_FILES = [
        f"/Users/christian/Documents/summer2023/MesoNet/data/{DATASET}_com_0.1-1Hz_35000/correlation.npy",
        f"/Users/christian/Documents/summer2023/MesoNet/data/{DATASET}_regions_0.1-1Hz_35000/correlation.npy",
        f"/Users/christian/Documents/summer2023/MesoNet/data/{DATASET}_comsquare_0.1-1Hz_35000/correlation.npy",
        f"/Users/christian/Documents/summer2023/MesoNet/data/{DATASET}_sopbilat_0.1-1Hz_35000/correlation.npy"
    ]
    MESONET_TAGS = [
        f"MesoNet_{DATASET}_com_35000",
        f"MesoNet_{DATASET}_regions_35000",
        f"MesoNet_{DATASET}_comsquare_35000",
        f"MesoNet_{DATASET}_sopbilat_35000",
    ]
    MATLAB_FILES = [
        f"/Users/christian/Documents/summer2023/matlab/my_data/{DATASET}/SOP_BilatRegionalCorr_35000.mat",
    ]
    MATLAB_TAGS = [
        f"SOP_BilatRegionalCorr_{DATASET}_35000",
    ]

    assert len(MESONET_FILES) == len(MESONET_TAGS)
    assert len(MATLAB_FILES) == len(MATLAB_TAGS)
    assert len(MATLAB_FILES) > 0

    n_regions = len(MATLAB)

    # Determine which regions appear in the MesoNet segmentation as well as the
    # MATLAB RHR regions.
    matlab_data = scipy.io.loadmat(MATLAB_FILES[0])
    matlab_labels = [label[0] for label in matlab_data["ROIlabels"][0]]
    matlab_region_numbers = []
    mesonet_region_numbers = []
    region_names = []
    for label in matlab_labels:
        if label in MATLAB_INVERSE:
            matlab_region_numbers.append(matlab_labels.index(label))
            mesonet_region_numbers.append(MATLAB_INVERSE[label])
            region_names.append(label)

    # Create the new correlation matrices containing only those regions that are
    # in both the MATLAB and MesoNet figures.
    mesonet_correlation_matrices = []
    for file in MESONET_FILES:
        matrix = np.load(file)
        matrix = matrix[mesonet_region_numbers, :][:, mesonet_region_numbers]
        mesonet_correlation_matrices.append(matrix)

    matlab_correlation_matrices = []
    for file in MATLAB_FILES:
        matrix = scipy.io.loadmat(file)
        matrix = matrix["corrmatrix1"].copy()
        matrix = matrix[matlab_region_numbers, :][:, matlab_region_numbers]
        matlab_correlation_matrices.append(matrix)

    correlation_matrices = (
        mesonet_correlation_matrices + matlab_correlation_matrices
    )
    correlation_tags = MESONET_TAGS + MATLAB_TAGS
    plot_labels = [f"{region_number} ({region_name})"
                   for region_number, region_name
                   in zip(mesonet_region_numbers, region_names)]
    n_datasets = len(correlation_matrices)

    # Plot and save.
    for minuend in range(0, n_datasets - 1):
        for subtrahend in range(minuend + 1, n_datasets):
            minuend_matrix = correlation_matrices[minuend]
            subtrahend_matrix = correlation_matrices[subtrahend]
            difference_matrix = minuend_matrix - subtrahend_matrix

            plt.rcParams.update({"font.size": 6})
            plot_title = f"{correlation_tags[minuend]} - {correlation_tags[subtrahend]}"
            plt.matshow(difference_matrix, vmin=-1.0, vmax=1.0)
            plt.title(plot_title)
            plt.xticks(range(n_regions), labels=plot_labels, rotation=45)
            plt.yticks(range(n_regions), labels=plot_labels)
            plt.tick_params(axis="x", labelbottom=True)
            plt.colorbar()
            plt.rcParams.update({"font.size": 3})
            for (i, j), difference in np.ndenumerate(difference_matrix):
                plt.text(j, i, f"{difference:0.3f}", ha="center", va="center")
            plt.savefig(os.path.join(SAVE_DIR, f"{plot_title}.png"), dpi=200)
            plt.clf()


def region_stds():
    DATASET = "awake1"
    MESONET_TIMECOURSE_FILES = [
        f"/Users/christian/Documents/summer2023/MesoNet/data/{DATASET}_com_0.1-1Hz_35000/timecourse.npy",
        f"/Users/christian/Documents/summer2023/MesoNet/data/{DATASET}_regions_0.1-1Hz_35000/timecourse.npy",
        f"/Users/christian/Documents/summer2023/MesoNet/data/{DATASET}_comsquare_0.1-1Hz_35000/timecourse.npy",
        f"/Users/christian/Documents/summer2023/MesoNet/data/{DATASET}_sopbilat_0.1-1Hz_35000/timecourse.npy",
    ]
    MESONET_TITLES = [
        "com",
        "regions",
        "comsquare",
        "sanity",
    ]
    MATLAB_TIMECOURSE_FILES = [
        f"/Users/christian/Documents/summer2023/matlab/my_data/{DATASET}/SOP_BilatRegionalCorr_35000.mat",
    ]
    MATLAB_TITLES = [
        "sop",
    ]

    assert len(MATLAB_TIMECOURSE_FILES) > 0
    assert len(MESONET_TIMECOURSE_FILES) == len(MESONET_TITLES)
    assert len(MATLAB_TIMECOURSE_FILES) == len(MATLAB_TITLES)

    matlab_data = scipy.io.loadmat(MATLAB_TIMECOURSE_FILES[0])
    matlab_labels = [label[0] for label in matlab_data["ROIlabels"][0]]
    matlab_region_numbers = []
    mesonet_region_numbers = []
    region_names = []
    for label in matlab_labels:
        if label in MATLAB_INVERSE:
            matlab_region_numbers.append(matlab_labels.index(label))
            mesonet_region_numbers.append(MATLAB_INVERSE[label])
            region_names.append(label)

    mesonet_timecourses = []
    for timecourse_file in MESONET_TIMECOURSE_FILES:
        timecourse = np.load(timecourse_file)
        timecourse = timecourse[mesonet_region_numbers, :]
        mesonet_timecourses.append(timecourse)

    matlab_timecourses = []
    for timecourse_file in MATLAB_TIMECOURSE_FILES:
        timecourse = scipy.io.loadmat(timecourse_file)
        timecourse = timecourse["timecourse"]
        timecourse = timecourse[matlab_region_numbers, :]
        matlab_timecourses.append(timecourse)

    timecourses = mesonet_timecourses + matlab_timecourses
    titles = MESONET_TITLES + MATLAB_TITLES
    stds = np.std(np.array(timecourses), axis=-1)

    def subcategorybar(labels, values, titles, width=0.8):
        n = len(values)
        labels_range = np.arange(len(labels))
        for i in range(n):
            plt.bar(labels_range - width / 2. + i / float(n) * width, values[i],
                    width=width / float(n), align="edge")
        plt.xticks(labels_range, labels)
        plt.legend(titles)

    subcategorybar(region_names, stds, titles)

    plt.title(DATASET)
    plt.show()


def _plot_correlation_matrix(
    correlation_matrix: np.array,
    region_points: Dict[Tuple[int, int], int],
    save_dir: str,
) -> np.array:
    # Save the original correlation matrix.
    np.save(os.path.join(save_dir, "correlation.npy"), correlation_matrix)

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
    plt.rcParams.update({"font.size": 6})
    plt.matshow(full_correlation_matrix)
    plt.xticks(range(len(sorted_regions)), labels=matrix_labels, rotation=45)
    plt.yticks(range(len(sorted_regions)), labels=matrix_labels)
    plt.tick_params(axis="x", labelbottom=True)
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "correlation.png"), dpi=200)
    plt.clf()

    # Save the full correlation matrix with upper right masked.
    masked_correlation_matrix = full_correlation_matrix * np.tri(len(sorted_regions)) * (1 - np.eye(len(sorted_regions)))
    plt.rcParams.update({"font.size": 6})
    plt.matshow(masked_correlation_matrix)
    plt.xticks(range(len(sorted_regions)), labels=matrix_labels, rotation=45)
    plt.yticks(range(len(sorted_regions)), labels=matrix_labels)
    plt.tick_params(axis="x", labelbottom=True)
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "correlation_masked.png"), dpi=200)
    plt.clf()

    # Save the full correlation matrix with higher values near the diagonal.
    reordered_matrix, new_order = reorder_matrix(full_correlation_matrix)
    plt.rcParams.update({"font.size": 6})
    plt.matshow(reordered_matrix)
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
    parser.add_argument("--still-image-file", type=str)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--mat-property", type=str, default=None)
    parser.add_argument("--mat-transpose-axes", type=int, nargs="+", default=None)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--n-frames", type=int, default=2000)
    parser.add_argument("--use-com", action='store_true')
    parser.add_argument("--square-com", action='store_true')
    parser.add_argument("--highlights", type=int, nargs="+", default=[])
    args = parser.parse_args()

    # fft(args)
    activity_complements(args)
    # activity(args)
    # seed_pixel_map(args)
    # correlation_matrix_comparison()
    # region_stds()



    # a1r = np.load('/Users/christian/Documents/summer2023/MesoNet/data/awake1_regions_preprocessed_0.1-1Hz/correlation.npy')
    # a1c = np.load('/Users/christian/Documents/summer2023/MesoNet/data/awake1_com_preprocessed_0.1-1Hz/correlation.npy')

    # a2r = np.load('/Users/christian/Documents/summer2023/MesoNet/data/awake2_regions_preprocessed_0.1-1Hz/correlation.npy')
    # a2c = np.load('/Users/christian/Documents/summer2023/MesoNet/data/awake2_com_preprocessed_0.1-1Hz/correlation.npy')

    # # m = a1r - a1c
    # # m = a1c - a1r
    # # m = a2r - a2c
    # m = a2c - a2r
    # plt.rcParams.update({"font.size": 6})
    # plt.matshow(m)
    # plt.xticks(range(m.shape[0]), labels=range(m.shape[0]), rotation=45)
    # plt.yticks(range(m.shape[0]), labels=range(m.shape[0]))
    # plt.tick_params(axis="x", labelbottom=True)
    # plt.colorbar()
    # plt.savefig(os.path.join("./data", "a2cpp-a2rpp.png"), dpi=200)
    # plt.clf()
