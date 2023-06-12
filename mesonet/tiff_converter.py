import argparse
import os

import numpy as np
import PIL.Image


# Modified from https://stackoverflow.com/questions/9627652/split-multi-page-tiff-with-python
class MultiImageTiff:
    def __init__(self, filename: str):
        self.image = PIL.Image.open(filename)
        self.image.seek(0)
        self.image_size = (self.image.tag[0x101][0], self.image.tag[0x100][0])
        self.current = self.image.tell()

    def get_frame(self, frame: int, map_to_255: bool = True) -> np.array:
        try:
            self.image.seek(frame)
        except EOFError:
            return None
        self.current = self.image.tell()
        image_array = self.image.getdata()
        if map_to_255:
            image_array = self._map_image_to_255(image_array)
        return np.reshape(image_array, self.image_size)

    def __iter__(self):
        self.image.seek(0)
        self.old = self.current
        self.current = self.image.tell()
        return self
    
    def next(self, map_to_255: bool = True) -> np.array:
        try:
            self.image.seek(self.current)
            self.current = self.image.tell() + 1
        except EOFError:
            self.image.seek(self.old)
            self.current = self.image.tell()
            raise StopIteration
        image_array = self.image.getdata()
        if map_to_255:
            image_array = self._map_image_to_255(image_array)
        return np.reshape(image_array, self.image_size)
    
    def _map_image_to_255(self, image: PIL.Image) -> np.array:
        image_array = np.array(image)
        image_min = np.amin(image_array)
        image_max = np.amax(image_array)
        mapped_image_array = np.interp(image_array,
                                       [image_min, image_max],
                                       [0, 255])
        return mapped_image_array.astype(np.uint8)


def main(args):
    tiff_image = MultiImageTiff(args.tiff_image_file)

    for image_to_save in args.images_to_save:
        image_array = tiff_image.get_frame(image_to_save)
        image = PIL.Image.fromarray(image_array)
        image.save(os.path.join(args.save_dir, f"{image_to_save}.png"))


if __name__ == "__main__":
    """
    python mesonet/tiff_converter.py \
        --tiff-image-file ./../matlab/my_data/04_awake_8x8_30hz_36500fr.tif \
        --images-to-save 0 6000 12000 18000 24000 \
        --save-dir ./mesonet_inputs/awake_data/atlas_brain
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff-image-file", type=str, required=True)
    parser.add_argument("--images-to-save", type=int, nargs="+", required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
