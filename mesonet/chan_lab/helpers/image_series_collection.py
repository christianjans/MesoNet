import dataclasses
from typing import Any, Dict, List

import numpy as np

from chan_lab.helpers.image_series import ImageSeries, ImageSeriesCreator


@dataclasses.dataclass(frozen=True)
class ImageSeriesCollectionArgs:
    file: str
    fps: float
    image_width: int
    image_height: int
    frame_index_offset: int
    kwargs: Dict[str, Any]


class ImageSeriesCollection:
    REFERENCE_INDEX = 0

    def __init__(self,
                 reference: ImageSeriesCollectionArgs,
                 *args: ImageSeriesCollectionArgs):
        assert reference.frame_index_offset == 0

        self._image_series: List[ImageSeries] = []
        self._offsets: List[int] = []
        self._fpses: List[float] = []

        args = list(args)
        args.insert(ImageSeriesCollection.REFERENCE_INDEX, reference)

        for arg in args:
            assert arg.frame_index_offset >= 0
            self._offsets.append(arg.frame_index_offset)

            assert arg.fps > 0
            self._fpses.append(arg.fps)

            image_series = ImageSeriesCreator.create_cached_image_series(
                    arg.file, arg.image_width, arg.image_height, "all",
                    **arg.kwargs)
            self._image_series.append(image_series)

    def get_frames(self, frame_index: int) -> List[np.ndarray]:
        assert frame_index >= max(self._offsets)

        images = []
        for i, image_series in enumerate(self._image_series):
            relative_frame_index = self._relative_frame_index(
                    frame_index, self._offsets[i], self._fpses[i])
            images.append(image_series.get_frame(relative_frame_index))

        return images

    def _relative_frame_index(self,
                              frame_index: int,
                              offset: int,
                              fps: float) -> int:
        reference_fps = self._fpses[ImageSeriesCollection.REFERENCE_INDEX]

        frame_time = frame_index / reference_fps
        start_time = offset / reference_fps
        diff_time = frame_time - start_time

        relative_frame_index = diff_time * fps
        relative_frame_index = int(round(relative_frame_index))

        return relative_frame_index
