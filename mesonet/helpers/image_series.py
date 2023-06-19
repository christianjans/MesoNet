from typing import Union

import numpy as np
import PIL.Image

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


class ImageSeriesCreator:
    @staticmethod
    def create(filename: str,
               image_width: int,
               image_height: int,
               n_frames: Union[int, str]) -> ImageSeries:
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
        else:
            raise ValueError(f"Unsupported image filename '{filename}'")
