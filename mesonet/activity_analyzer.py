import argparse
import os
import pickle
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from tiff_image import MultiImageTiff

REGION_POINTS_WIDTH_MAX = 512
REGION_POINTS_HEIGHT_MAX = 512


class MasksManager:
    def __init__(self,
                 region_points_file: str,
                 image_width: int,
                 image_height: int):
        self.image_width = image_width
        self.image_height = image_height

        assert REGION_POINTS_WIDTH_MAX % self.image_width == 0
        assert REGION_POINTS_HEIGHT_MAX % self.image_height == 0

        # Factor to go from region points to image points.
        self.scale_factor_x = self.image_width / REGION_POINTS_WIDTH_MAX
        self.scale_factor_y = self.image_height / REGION_POINTS_HEIGHT_MAX

        with open(region_points_file, 'rb') as f:
            self.region_points: Dict[Tuple[int, int], int] = pickle.load(f)

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
        return len({region for region in self.region_points.values()})


def fft(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    masks_manager = MasksManager(args.region_points_file,
                                 args.image_width,
                                 args.image_height)
    tiff_images = MultiImageTiff(args.tiff_image_file)

    fft_values = [[] for _ in range(masks_manager.n_regions)]

    for i, image in enumerate(tiff_images):
        if i == args.n_frames:
            break

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
    tiff_images = MultiImageTiff(args.tiff_image_file)

    data = [[[], []] for _ in range(len(masks_manager.masks) // 2)]

    for i, image in enumerate(tiff_images):
        if i == args.n_frames:
            break

        masked_image = image * masks_manager.masks
        masked_sums = masked_image.sum(axis=(1, 2)) / masks_manager.masks.sum(axis=(1, 2))

        for label in range(len(masks_manager.masks) // 2):
            complement_label = len(masks_manager.masks) - label - 1
            data[label][0].append(masked_sums[label])
            data[label][1].append(masked_sums[complement_label])

    for i, values in enumerate(data):
        label = i
        complement_label = len(masks_manager.masks) - label - 1
        plot_filename = f"complement_{label}-{complement_label}.png"

        plt.title(f"{label} - {complement_label} activity")
        plt.plot(values[0], label=label)
        plt.plot(values[1], label=complement_label)
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, plot_filename))
        plt.clf()


def activity(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    masks_manager = MasksManager(args.region_points_file,
                                 args.image_width,
                                 args.image_height)
    tiff_images = MultiImageTiff(args.tiff_image_file)

    plt.imshow(masks_manager.masks[0])
    plt.show()

    data = [[] for _ in range(len(masks_manager.masks))]

    for i, image in enumerate(tiff_images):
        if i == args.n_frames:
            break

        masked_image = image * masks_manager.masks
        masked_sums = masked_image.sum(axis=(1, 2)) / masks_manager.masks.sum(axis=(1, 2))

        for j in range(len(masks_manager.masks)):
            data[j].append(masked_sums[j])

    for i, values in enumerate(data):
        plt.plot(values)
    plt.savefig(os.path.join(args.save_dir, f"activity.png"))
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-points-file", type=str, required=True)
    parser.add_argument("--tiff-image-file", type=str, required=True)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--n-frames", type=int, default=2000)
    args = parser.parse_args()

    # fft(args)
    activity_complements(args)
    # activity(args)
