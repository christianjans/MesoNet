import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import argparse
import os

import numpy as np
import PIL
from PIL import ImageEnhance

from mesonet.chan_lab.helpers.image_series import ImageSeriesCreator
from mesonet.chan_lab.helpers.utils import config_to_namespace


def save_images(args: argparse.Namespace):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    image_series = ImageSeriesCreator.create_cached_image_series(
            args.image_file, args.image_width, args.image_height, "all")

    for image_to_save in args.images_to_save:
        image_array = image_series.get_frame(image_to_save)
        image_min, image_max = np.amin(image_array), np.amax(image_array)
        image_array = np.interp(image_array, [image_min, image_max], [0, 255])
        image_array = image_array.astype(np.uint8)

        image_array = np.pad(image_array, (args.padding,), constant_values=0)

        image = PIL.Image.fromarray(image_array)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(args.brightness)

        print((
            f"Saving image {image_to_save} with min of {np.min(image_array)} "
            f"and max of {np.max(image_array)} and {image_array.dtype} data "
            f"type"
        ))

        image.save(os.path.join(args.save_dir, f"{image_to_save}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_args = config_to_namespace(args.config)

    save_images(config_args)
