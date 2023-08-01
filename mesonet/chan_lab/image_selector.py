import argparse
import os

import numpy as np
import PIL

from chan_lab.helpers.image_series import ImageSeriesCreator


def save_images(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    image_series = ImageSeriesCreator.create_cached_image_series(
            args.image_file, args.image_width, args.image_height, "all")

    for image_to_save in args.images_to_save:
        image_array = image_series.get_frame(image_to_save)
        image_min, image_max = np.amin(image_array), np.amax(image_array)
        image_array = np.interp(image_array, [image_min, image_max], [0, 255])
        image_array = image_array.astype(np.uint8)
        print(np.max(image_array))
        print(np.min(image_array))
        print(image_array.dtype)
        image_array = np.pad(image_array, (args.padding,), constant_values=0)
        image = PIL.Image.fromarray(image_array)
        image.save(os.path.join(args.save_dir, f"{image_to_save}.png"))


if __name__ == "__main__":
    """
    python mesonet/chan_lab/image_selector.py \
        --image-file ./../matlab/my_data/04_awake_8x8_30hz_36500fr.tif \
        --images-to-save 0 6000 12000 18000 24000 \
        --save-dir ./mesonet_inputs/awake_data/atlas_brain
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--images-to-save", type=int, nargs="+", required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    args = parser.parse_args()
    save_images(args)
