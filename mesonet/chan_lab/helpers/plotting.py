import dataclasses
from typing import Any, Dict, Iterable

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import scipy

from helpers.image_series import ImageSeriesCreator


@dataclasses.dataclass(frozen=True)
class PlotterArgs:
    filename: str
    fps: float
    offset: int
    title: str

    def __new__(cls, *args, **kwargs):
        if cls == PlotterArgs:
            raise TypeError("Cannot instantiate abstract class")
        return super().__new__(cls)


@dataclasses.dataclass(frozen=True)
class PupillometryPlotterArgs(PlotterArgs):
    pass


@dataclasses.dataclass(frozen=True)
class ImagePlotterArgs(PlotterArgs):
    image_width: int
    image_height: int
    kwargs: Dict[str, Any]


@dataclasses.dataclass(frozen=True)
class VideoPlotterArgs(PlotterArgs):
    pass


class SeriesPlotter:
    def __init__(self, filename: str, fps: float, offset: int, title: str):
        self._filename = filename
        self._fps = fps
        self._offset = offset
        self._title = title

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def title(self) -> str:
        return self._title

    @property
    def n_frames(self) -> int:
        raise NotImplementedError

    def update(self, frame_index: int, figure: Figure, axes: Axes):
        raise NotImplementedError


class PupillometryPlotter(SeriesPlotter):
    def __init__(self, args: PupillometryPlotterArgs):
        super().__init__(args.filename, args.fps, args.offset, args.title)
        self._data = scipy.io.loadmat(self.filename)["R"]

    @property
    def n_frames(self) -> int:
        return int(max(self._data[:, 0]))

    def update(self, frame_index: int, figure: Figure, axes: Axes):
        print(f"pupillometry = {int(self._data[frame_index, 0])}")
        frame = self._data[frame_index, 0]
        left_x, right_x = frame - 200, frame + 200
        axes.clear()
        axes.set_title(self.title)
        axes.set_xlim(left_x, right_x)
        axes.plot(self._data[:, 0], self._data[:, 1])
        axes.axvline(self._data[frame_index, 0])


class ImagePlotter(SeriesPlotter):
    def __init__(self, args: ImagePlotterArgs):
        super().__init__(args.filename, args.fps, args.offset, args.title)
        self._image_series = \
                ImageSeriesCreator.create_cached_image_series(self.filename,
                                                              args.image_width,
                                                              args.image_height,
                                                              n_frames="all",
                                                              **args.kwargs)

    @property
    def n_frames(self) -> int:
        return self._image_series.n_frames

    def update(self, frame_index: int, figure: Figure, axes: Axes):
        frame = self._image_series.get_frame(frame_index)

        axes.clear()
        axes.set_title(self.title)
        axes.imshow(frame)


class VideoPlotter(SeriesPlotter):
    def __init__(self, args: VideoPlotterArgs):
        super().__init__(args.filename, args.fps, args.offset, args.title)
        self._image_series = \
                ImageSeriesCreator.create_uncached_image_series(args.filename)

    @property
    def n_frames(self) -> int:
        return self._image_series.n_frames

    def update(self, frame_index: int, figure: Figure, axes: Axes):
        print(f"video = {frame_index}")
        frame = self._image_series.get_frame(frame_index)

        axes.clear()
        axes.set_title(self.title)
        axes.imshow(frame)


def _plotter_from_args(args: PlotterArgs) -> SeriesPlotter:
    if isinstance(args, PupillometryPlotterArgs):
        return PupillometryPlotter(args)
    elif isinstance(args, ImagePlotterArgs):
        return ImagePlotter(args)
    elif isinstance(args, VideoPlotterArgs):
        return VideoPlotter(args)
    else:
        return ValueError(f"Invalid arguments: {args}")


class PlotterCollection:
    def __init__(self,
                 figure: Figure,
                 plotter_args: Iterable[PlotterArgs],
                 rows: int):
        self._figure = figure
        self._n_plots = len(plotter_args)
        self._n_rows = rows
        self._n_columns = (
            self._n_plots // self._n_rows if self._n_plots % self._n_rows == 0
            else self._n_plots // self._n_rows + 1
        )
        self._plots = [
            self._figure.add_subplot(self._n_rows, self._n_columns, i + 1)
            for i in range(self._n_plots)
        ]
        self._plotters = [_plotter_from_args(args) for args in plotter_args]

        self._reference_offset = self._plotters[0].offset
        self._reference_fps = self._plotters[0].fps
        all_offsets = [plotter.offset for plotter in self._plotters]
        all_max_frames = [
            offset + plotter.n_frames - 1
            for offset, plotter in zip(all_offsets, self._plotters)
        ]
        assert self._reference_offset <= min(all_offsets)

        self._min_frame_index = int(max(all_offsets))
        self._max_frame_index = int(min(all_max_frames))

    @property
    def min_frame_index(self) -> int:
        return self._min_frame_index

    @property
    def max_frame_index(self) -> int:
        return self._max_frame_index

    def update_plots(self, frame_index: int):
        assert frame_index >= self._min_frame_index
        assert frame_index <= self._max_frame_index

        for axes, plotter in zip(self._plots, self._plotters):
            relative_frame_index = self._relative_frame_index(frame_index,
                                                              plotter.offset,
                                                              plotter.fps)
            plotter.update(relative_frame_index, self._figure, axes)

    def _relative_frame_index(self,
                              frame_index: int,
                              offset: int,
                              fps: float) -> int:
        reference_time = self._reference_offset / self._reference_fps
        frame_time = frame_index / self._reference_fps
        start_time = offset / self._reference_fps
        diff_time = reference_time + frame_time - start_time

        relative_frame_index = diff_time * fps
        relative_frame_index = int(round(relative_frame_index))

        return relative_frame_index
