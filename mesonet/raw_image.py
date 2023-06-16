"""
Load raw images saved in MATLAB to NumPy arrays.
"""

import argparse
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

BIG_ENDIAN_F32 = ">f4"


class RawImage:
    def __init__(self,
                 filename: str,
                 image_width: int,
                 image_height: int,
                 n_frames: Union[int, str] = "all",
                 dtype: str = BIG_ENDIAN_F32):
        self.image = np.fromfile(filename, dtype=dtype)
        self.image = np.reshape(self.image, (-1, image_height, image_width))

        if isinstance(n_frames, int):
            self.image = self.image[:n_frames]

        self.current_image_index = 0
        self.max_image_index = len(self.image)

    def __iter__(self):
        self.current_image_index = 0
        return self

    def __next__(self):
        if self.current_image_index < self.max_image_index:
            current_image = self.image[self.current_image_index]
            self.current_image_index += 1
            return current_image
        else:
            raise StopIteration


def main(args):
    raw_image = RawImage(args.raw_filename, args.image_width, args.image_height, n_frames=10)
    for im in raw_image:
        plt.imshow(im)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=str, required=True)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--dtype", type=str, default=BIG_ENDIAN_F32)
    args = parser.parse_args()

    main(args)
