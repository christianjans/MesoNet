import dataclasses
from typing import Any, Dict, Iterable, Tuple, Union

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import scipy

from mesonet.chan_lab.helpers.image_series import ImageSeriesCreator
from mesonet.chan_lab.activity_analyzer import MasksManager


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
    region_points: str = None


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

    def update(self, frame_index: int, figure: Figure, axes: Axes,
               started: bool):
        raise NotImplementedError


class PupillometryPlotter(SeriesPlotter):
    def __init__(self, args: PupillometryPlotterArgs):
        super().__init__(args.filename, args.fps, args.offset, args.title)
        self._data = scipy.io.loadmat(self.filename)["R"]
        self._axes_data = None

    @property
    def n_frames(self) -> int:
        return int(max(self._data[:, 0]))

    def update(self, frame_index: int, figure: Figure, axes: Axes,
               started: bool):
        print(f"{self.title} frame: {int(self._data[frame_index, 0])}")
        WINDOW = 100
        visible_x = self._data[(frame_index - WINDOW):(frame_index + WINDOW + 1), 0]
        visible_y = self._data[(frame_index - WINDOW):(frame_index + WINDOW + 1), 1]
        # start_frame, end_frame = visible_x[0], visible_x[-1]
        # left_delta, right_delta = int(start_frame - (frame_index + 1)), int(end_frame - (frame_index + 1))

        # print(f"left_delta = {left_delta}, right_delta = {right_delta}")
        left, right = int(frame_index - WINDOW), int(frame_index + WINDOW + 1)
        x = [int(2 * i) for i in range(left - frame_index, right - frame_index)]

        if not started:
            self._axes_data = axes.plot([])[0]
            axes.set_title(self.title)
            axes.set_xlabel("Relative frame", fontdict={"size": 80})
            axes.tick_params(axis="both", labelsize=80)
            axes.axvline(self._data[frame_index, 0])
            axes.set_xlim(visible_x[0], visible_x[-1])
            axes.set_ylim(np.min(self._data[:, 1]), np.max(self._data[:, 1]))
            axes.set_xticks(visible_x[::10])
            axes.set_xticklabels(x[::10])
        else:
            self._axes_data.set_data(self._data[:, 0], self._data[:, 1])
            axes.draw_artist(self._axes_data)
            axes.axvline(self._data[frame_index, 0])
            axes.set_xlim(visible_x[0], visible_x[-1])
            axes.set_ylim(np.min(self._data[:, 1]), np.max(self._data[:, 1]))


class ImagePlotter(SeriesPlotter):
    def __init__(self, args: ImagePlotterArgs):
        super().__init__(args.filename, args.fps, args.offset, args.title)
        self._image_series = \
                ImageSeriesCreator.create_cached_image_series(self.filename,
                                                              args.image_width,
                                                              args.image_height,
                                                              n_frames="all",
                                                              **args.kwargs)
        self._axes_image = None

        self._mask = None
        if args.region_points:
            self._mask = MasksManager(args.region_points,
                                      args.image_width,
                                      args.image_height)
            self._mask = self._mask.masks
            self._mask = np.logical_or.reduce(self._mask, axis=0)
            self._mask = np.ma.masked_where(self._mask == 0, self._mask)

    @property
    def n_frames(self) -> int:
        return self._image_series.n_frames

    def update(self, frame_index: int, figure: Figure, axes: Axes,
               started: bool):
        print(f"{self.title} frame: {frame_index}")
        frame = self._image_series.get_frame(frame_index)

        if not started:
            self._axes_image = axes.imshow(np.zeros_like(frame),
                                           vmin=-1, vmax=1)
            if self._mask is not None:
                self._axes_image = axes.imshow(self._mask, alpha=0.8, vmin=-1,
                                            vmax=1)
            axes.set_title(self.title)
        else:
            self._axes_image.set_data(frame)
            axes.draw_artist(self._axes_image)

        # axes.clear()
        # axes.set_title(self.title)
        # axes.imshow(frame)


class VideoPlotter(SeriesPlotter):
    def __init__(self, args: VideoPlotterArgs):
        super().__init__(args.filename, args.fps, args.offset, args.title)
        self._image_series = \
                ImageSeriesCreator.create_uncached_image_series(args.filename)
        self._axes_image = None

    @property
    def n_frames(self) -> int:
        return self._image_series.n_frames

    def update(self, frame_index: int, figure: Figure, axes: Axes,
               started: bool):
        print(f"{self.title} frame: {frame_index}")
        frame = self._image_series.get_frame(frame_index)

        if not started:
            self._axes_image = axes.imshow(np.zeros_like(frame))
            axes.set_title(self.title)
        else:
            self._axes_image.set_data(frame)
            axes.draw_artist(self._axes_image)

        # axes.clear()
        # axes.set_title(self.title)
        # axes.imshow(frame)


def _plotter_from_args(args: PlotterArgs) -> SeriesPlotter:
    if isinstance(args, PupillometryPlotterArgs):
        return PupillometryPlotter(args)
    elif isinstance(args, ImagePlotterArgs):
        return ImagePlotter(args)
    elif isinstance(args, VideoPlotterArgs):
        return VideoPlotter(args)
    else:
        raise ValueError(f"Invalid arguments: {args}")


def args_from_yaml(yaml_dict: Dict[str, Any]) -> PlotterArgs:
    if yaml_dict["type"] == "video":
        return VideoPlotterArgs(**yaml_dict["args"])
    elif yaml_dict["type"] == "image":
        return ImagePlotterArgs(**yaml_dict["args"])
    elif yaml_dict["type"] == "pupillometry":
        return PupillometryPlotterArgs(**yaml_dict["args"])
    else:
        raise ValueError(f"Invalid type: `{yaml_dict['type']}`")


class PlotterCollection:
    PLOT_FONT_SIZE = 80

    def __init__(self,
                 canvas: FigureCanvasTkAgg,
                 figure: Figure,
                 plotter_args: Iterable[PlotterArgs],
                 rows: int):
        matplotlib.rcParams.update({
            "font.size": PlotterCollection.PLOT_FONT_SIZE
        })
        self._canvas = canvas
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

        # Used for blitting.
        self._started = False
        self._plot_backgrounds = []

        self._figure.subplots_adjust(wspace=0.1, hspace=0.1)

    @property
    def min_frame_index(self) -> int:
        return self._min_frame_index

    @property
    def max_frame_index(self) -> int:
        return self._max_frame_index

    def update_plots(self, frame_index: int):
        assert frame_index >= self._min_frame_index
        assert frame_index <= self._max_frame_index

        if not self._started:
            for axes, plotter in zip(self._plots, self._plotters):
                relative_frame_index = self._relative_frame_index(frame_index,
                                                                plotter.offset,
                                                                plotter.fps)
                plotter.update(relative_frame_index,
                               self._figure,
                               axes,
                               started=self._started)

            self._canvas.draw()
            for axes in self._plots:
                self._plot_backgrounds.append(
                        self._canvas.copy_from_bbox(axes.bbox))
            self._started = True
        else:
            for background in self._plot_backgrounds:
                self._canvas.restore_region(background)

            for axes, plotter in zip(self._plots, self._plotters):
                relative_frame_index = self._relative_frame_index(frame_index,
                                                                plotter.offset,
                                                                plotter.fps)
                plotter.update(relative_frame_index,
                               self._figure,
                               axes,
                               started=self._started)

            for axes in self._plots:
                self._canvas.blit(axes.bbox)

            self._canvas.flush_events()

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
