import argparse
import os
from typing import List, Union

import numpy as np
import PIL.Image
import scipy

BIG_ENDIAN_F32 = ">f4"


class ImageSeries:
    def __init__(self,
                 filename: str,
                 image_width: int,
                 image_height: int,
                 n_frames: Union[int, str] = "all"):
        self._filename = filename
        self._image_width = image_width
        self._image_height = image_height
        self._n_frames = n_frames
        self._image_array = self._load_image_series(self._filename,
                                                    self._image_width,
                                                    self._image_height,
                                                    self._n_frames)
        self._current_image_index = 0
        self._max_image_index = len(self._image_array)

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def image_array(self) -> np.ndarray:
        return self._image_array

    def __iter__(self):
        self._current_image_index = 0
        return self

    def __next__(self) -> np.ndarray:
        if self._current_image_index < self._max_image_index:
            current_image = self._image_array[self._current_image_index]
            self._current_image_index += 1
            return current_image
        else:
            raise StopIteration

    def get_frame(self, frame_index: int) -> np.ndarray:
        return self._image_array[frame_index]

    def _load_image_series(self,
                           filename: str,
                           image_width: int,
                           image_height: int,
                           n_frames: Union[str, int]) -> np.ndarray:
        raise NotImplementedError


class RawImageSeries(ImageSeries):
    def __init__(self,
                 filename: str,
                 image_width: int,
                 image_height: int,
                 n_frames: Union[int, str] = "all"):
        super().__init__(filename, image_width, image_height, n_frames)

    def _load_image_series(self,
                           filename: str,
                           image_width: int,
                           image_height: int,
                           n_frames: Union[int, str]) -> np.ndarray:
        image_array = np.fromfile(filename, dtype=BIG_ENDIAN_F32)
        image_array = np.reshape(image_array,
                                 (-1, image_height, image_width))

        if isinstance(n_frames, int):
            image_array = image_array[:n_frames]

        return image_array


# Modified from https://stackoverflow.com/questions/9627652/split-multi-page-tiff-with-python
class TiffImageSeries(ImageSeries):
    def __init__(self,
                 filename: str,
                 image_width: int,
                 image_height: int,
                 n_frames: Union[int, str] = "all"):
        super().__init__(filename, image_width, image_height, n_frames)

    def _load_image_series(self,
                           filename: str,
                           image_width: int,
                           image_height: int,
                           n_frames: Union[int, str]) -> np.ndarray:
        image = PIL.Image.open(filename)
        image.seek(0)
        current_ptr = image.tell()

        image_size = (image.tag[0x101][0], image.tag[0x100][0])
        assert image_size == (image_height, image_width)

        images = []

        while True:
            try:
                image.seek(current_ptr)
                current_ptr = image.tell() + 1
            except EOFError:
                break
            images.append(image.getdata())

        if isinstance(n_frames, int):
            images = images[:n_frames]

        image_array = np.stack(images, axis=0)
        image_array = np.reshape(image_array,
                                 (-1, image_size[0], image_size[1]))

        return image_array


class MatImageSeries(ImageSeries):
    def __init__(self,
                 filename: str,
                 image_width: int,
                 image_height: int,
                 property: str,
                 transpose_axes: List,
                 n_frames: Union[int, str] = "all"):
        self._property = property
        self._transpose_axes = transpose_axes
        super().__init__(filename, image_width, image_height, n_frames)

    def _load_image_series(self,
                           filename: str,
                           image_width: int,
                           image_height: int,
                           n_frames: Union[int, str]) -> np.ndarray:
        image_array = scipy.io.loadmat(filename)[self._property]
        image_array = np.transpose(image_array, axes=self._transpose_axes)

        if isinstance(n_frames, int):
            image_array = image_array[:n_frames]

        return image_array


class ImageSeriesCreator:
    @staticmethod
    def create(filename: str,
               image_width: int,
               image_height: int,
               n_frames: Union[int, str],
               **kwargs) -> ImageSeries:
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            return TiffImageSeries(filename,
                                   image_width,
                                   image_height,
                                   n_frames)
        elif filename.endswith(".raw"):
            return RawImageSeries(filename,
                                  image_width,
                                  image_height,
                                  n_frames)
        elif filename.endswith(".mat"):
            return MatImageSeries(filename=filename,
                                  image_width=image_width,
                                  image_height=image_height,
                                  n_frames=n_frames,
                                  **kwargs)
        else:
            raise ValueError(f"Unsupported image filename '{filename}'")


def save_images(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    image_series = ImageSeriesCreator.create(args.image_file,
                                             args.image_width,
                                             args.image_height,
                                             "all")

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
    python mesonet/tiff_converter.py \
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
