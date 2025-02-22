import dataclasses
from typing import Any, Dict, Iterable, List

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import scipy

from mesonet.chan_lab.activity_analyzer import MasksManager
from mesonet.chan_lab.helpers.event_frames import EventFrames
from mesonet.chan_lab.helpers.image_series import ImageSeriesCreator


@dataclasses.dataclass(frozen=True)
class PlotterArgs:
    filename: str
    event_frames: Iterable[int]
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
    def __init__(self, filename: str, event_frames: Iterable[int], title: str):
        self._filename = filename
        self._event_frames = EventFrames(event_frames)
        self._title = title

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def event_frames(self) -> EventFrames:
        return self._event_frames

    @property
    def title(self) -> str:
        return self._title

    @property
    def n_frames(self) -> int:
        raise NotImplementedError

    def data_segment(self,
                     frame_index: int,
                     left: int = 0,
                     right: int = 0,
                     every: int = 1) -> np.ndarray:
        raise NotImplementedError

    def update(self, frame_index: int, figure: Figure, axes: Axes,
               started: bool):
        raise NotImplementedError


class PupillometryPlotter(SeriesPlotter):
    WINDOW = 100

    def __init__(self, args: PupillometryPlotterArgs):
        super().__init__(args.filename, args.event_frames, args.title)
        self._data = scipy.io.loadmat(self.filename)["R"]
        self._start_frame = self._data[0, 0]
        self._delta_frame = self._data[1, 0] - self._start_frame
        self._axes_data = None

    @property
    def n_frames(self) -> int:
        return int(max(self._data[:, 0]))

    def data_segment(self,
                     frame_index: int,
                     left: int = 0,
                     right: int = 0,
                     every: int = 1) -> np.ndarray:
        left_index = int((frame_index - left * every) // self._delta_frame)
        right_index = int((frame_index + right * every) // self._delta_frame)

        frames = [
            np.array([self._data[i, 0], self._data[i, 1]])
            for i in range(left_index, right_index + 1, every)
        ]
        frames = np.array(frames)
        frames = frames.T
        return frames

    def update(self, frame_index: int, figure: Figure, axes: Axes,
               started: bool):
        frame_index = int(frame_index // self._delta_frame)
        print(f"{self.title} frame: {int(self._data[frame_index, 0])}")
        visible_x = self._data[(frame_index - PupillometryPlotter.WINDOW):(frame_index + PupillometryPlotter.WINDOW + 1), 0]
        # start_frame, end_frame = visible_x[0], visible_x[-1]
        # left_delta, right_delta = int(start_frame - (frame_index + 1)), int(end_frame - (frame_index + 1))

        # print(f"left_delta = {left_delta}, right_delta = {right_delta}")
        left = int(frame_index - PupillometryPlotter.WINDOW)
        right = int(frame_index + PupillometryPlotter.WINDOW)
        x = [int(self._delta_frame * i)
             for i in range(left - frame_index, right - frame_index + 1)]

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
        super().__init__(args.filename, args.event_frames, args.title)
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

    def data_segment(self,
                     frame_index: int,
                     left: int = 0,
                     right: int = 0,
                     every: int = 1) -> np.ndarray:
        left_index = frame_index - left * every
        right_index = frame_index + right * every

        frames = [
            self._image_series.get_frame(i)
            for i in range(left_index, right_index + 1, every)
        ]
        frames = np.array(frames)
        return frames

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
        super().__init__(args.filename, args.event_frames, args.title)
        self._image_series = \
                ImageSeriesCreator.create_uncached_image_series(args.filename)
        self._axes_image = None

    @property
    def n_frames(self) -> int:
        return self._image_series.n_frames

    def data_segment(self,
                     frame_index: int,
                     left: int = 0,
                     right: int = 0,
                     every: int = 1) -> np.ndarray:
        left_index = frame_index - left * every
        right_index = frame_index + right * every

        frames = [
            self._image_series.get_frame(i)
            for i in range(left_index, right_index + 1, every)
        ]
        frames = np.array(frames)
        return frames

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


class Collection:
    PLOT_FONT_SIZE = 80

    def __init__(self, plotter_args: Iterable[PlotterArgs]):
        self._plotters = [_plotter_from_args(args) for args in plotter_args]
        self._reference_event_frames = self._plotters[0].event_frames
        self._min_frame_index = self._reference_event_frames.min
        self._max_frame_index = self._reference_event_frames.max

    @property
    def min_frame_index(self) -> int:
        return self._min_frame_index
    
    @property
    def max_frame_index(self) -> int:
        return self._max_frame_index

    def update(self, frame_index: int, *args, **kwargs) -> Any:
        raise NotImplementedError

    def _relative_frame_index(self,
                              frame_index: int,
                              event_frames: EventFrames) -> int:
        return event_frames.equivalent_frame(
                frame_index, self._reference_event_frames)


class EventHighlighter(Collection):
    PLOT_FONT_SIZE = 80

    def __init__(self, plotter_args: Iterable[PlotterArgs]):
       super().__init__(plotter_args)

    @property
    def min_frame_index(self) -> int:
        return self._min_frame_index

    @property
    def max_frame_index(self) -> int:
        return self._max_frame_index

    def update(self,
               frame_index: int,
               left: int = 0,
               right: int = 0,
               every: int = 1) -> Any:
        assert frame_index >= self._min_frame_index, (
            f"frame {frame_index} must be >= min frame index "
            f"{self._min_frame_index}"
        )
        assert frame_index <= self._max_frame_index, (
            f"frame {frame_index} must be <= max frame index "
            f"{self._max_frame_index}"
        )

        frames = []
        for plotter in self._plotters:
            equivalent_frame = self._relative_frame_index(
                    frame_index, plotter.event_frames)
            frames.append(plotter.data_segment(equivalent_frame,
                                               left,
                                               right,
                                               every))

        return frames


class PlotterCollection(Collection):
    PLOT_FONT_SIZE = 80

    def __init__(self,
                 canvas: FigureCanvasTkAgg,
                 figure: Figure,
                 plotter_args: Iterable[PlotterArgs],
                 rows: int):
        super().__init__(plotter_args)

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
        self._figure.subplots_adjust(wspace=0.1, hspace=0.1)

        # Used for blitting.
        self._started = False
        self._plot_backgrounds = []

    def update(self, frame_index: int) -> Any:
        assert frame_index >= self._min_frame_index
        assert frame_index <= self._max_frame_index

        if not self._started:
            for axes, plotter in zip(self._plots, self._plotters):
                relative_frame_index = self._relative_frame_index(
                        frame_index, plotter.event_frames)
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
                relative_frame_index = self._relative_frame_index(
                        frame_index, plotter.event_frames)
                plotter.update(relative_frame_index,
                               self._figure,
                               axes,
                               started=self._started)

            for axes in self._plots:
                self._canvas.blit(axes.bbox)

            self._canvas.flush_events()
