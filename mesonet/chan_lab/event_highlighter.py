import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from mesonet.chan_lab.helpers.utils import config_to_namespace
from mesonet.chan_lab.helpers.plotting import (
    args_from_yaml, PlotterCollection
)


params = {
    # 'legend.fontsize': 'x-large',
    # 'figure.figsize': (15, 5),
    # 'axes.labelsize': 'x-large',
    'axes.titlesize': 'xx-small',
    # 'xtick.labelsize': 'x-large',
    # 'ytick.labelsize': 'x-large'
}
plt.rcParams.update(params)


def _plot_pupil(
        data: np.ndarray, left: int, every: int, frames_of_interest: List[int]):
    rows = data.shape[0]
    columns = data.shape[1]
    figure = plt.figure()

    for i in range(rows):
        for j in range(columns):
            relative_frame = every * (j - left)
            title = f"{frames_of_interest[i]}: {relative_frame}"
            axes = figure.add_subplot(rows, columns, i * columns + j + 1)
            axes.set_axis_off()
            axes.set_title(title)
            axes.imshow(data[i][j], vmin=-1.0, vmax=1.0)

    figure.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.05, hspace=0.0)


def _plot_body(
        data: np.ndarray, left: int, every: int, frames_of_interest: List[int]):
    rows = data.shape[0]
    columns = data.shape[1]
    figure = plt.figure()

    for i in range(rows):
        for j in range(columns):
            relative_frame = every * (j - left)
            title = f"{frames_of_interest[i]}: {relative_frame}"
            axes = figure.add_subplot(rows, columns, i * columns + j + 1)
            axes.set_axis_off()
            axes.set_title(title)
            axes.imshow(data[i][j], vmin=-1.0, vmax=1.0)

    figure.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.05, hspace=0.0)


def _plot_mesoscale(
        data: np.ndarray, left: int, every: int, frames_of_interest: List[int]):
    """The data argument is a NumPy array with shape

    (
        number of frames of interest,
        left + right + 1,
        image height,
        image width
    )
    """
    rows = data.shape[0]
    columns = data.shape[1]
    # figure, axes = plt.subplots(rows, columns, sharex=True)
    figure = plt.figure()
    figure_aggregated = plt.figure()

    for i in range(rows):
        for j in range(columns):
            relative_frame = every * (j - left)
            title = f"{frames_of_interest[i]}: {relative_frame}"
            axes = figure.add_subplot(rows, columns, i * columns + j + 1)
            axes.set_axis_off()
            axes.set_title(title)
            axes.imshow(data[i][j], vmin=-1.0, vmax=1.0)
            # axes[i, j].set_axis_off()
            # axes[i, j].set_title(title)
            # axes[i, j].imshow(data[i][j])

    mean_data = np.mean(data, axis=0)
    for i, mean_image in enumerate(mean_data):
        relative_frame = every * (i - left)
        axes = figure_aggregated.add_subplot(1, len(mean_data), i + 1)
        axes.set_axis_off()
        axes.set_title(f"{relative_frame}")
        axes.imshow(mean_image, vmin=-1.0, vmax=1.0)

    figure.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.05, hspace=0.0)
    figure_aggregated.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.05, hspace=0.0)


def _plot_pupillometry(
        data: np.ndarray, left: int, every: int, frames_of_interest: List[int]):
    """The data argument is a NumPy array with shape

    (
        number of frames of interest,
        2,
        left + right + 1
    )

    where 2 represents the x and y coordinates.
    """
    print(data.shape)
    columns = data.shape[0]
    points = data.shape[-1]
    figure_individuals = plt.figure()
    figure_aggregated = plt.figure()

    x_data = data[:, 0, :]
    y_data = data[:, 1, :]
    delta_x =( x_data[0, 1] - x_data[0, 0]) // every
    x_ticks = [every * (i * delta_x - left) for i in range(points)]

    for i, (y, frame_of_interest) in enumerate(zip(y_data, frames_of_interest)):
        axes = figure_individuals.add_subplot(1, columns, i + 1)
        axes.plot(x_ticks, y)
        axes.set_title(f"{frame_of_interest}")

    mean = np.mean(y_data, axis=0)
    std = np.std(y_data, axis=0)
    axes = figure_aggregated.add_subplot()
    axes.plot(x_ticks, mean)
    axes.fill_between(x_ticks, mean - std, mean + std, alpha=0.3)


def main(args: argparse.Namespace):
    pupil_args = args_from_yaml(args.pupil)
    body_args = args_from_yaml(args.body)
    mesoscale_args = args_from_yaml(args.mesoscale)
    pupillometry_args = args_from_yaml(args.pupillometry)

    collection = PlotterCollection(
            canvas=None, figure=None,
            plotter_args=[pupil_args, body_args, mesoscale_args,
                          pupillometry_args],
            rows=1)

    pupil_data = []
    body_data = []
    mesoscale_data = []
    pupillometry_data = []

    for frame_of_interest in args.frames_of_interest:
        data_segments = collection.data_segments(
                frame_of_interest, args.frames_left, args.frames_right,
                args.skip_every)
        
        pupil_data.append(data_segments[0])
        body_data.append(data_segments[1])
        mesoscale_data.append(data_segments[2])
        pupillometry_data.append(data_segments[3])

    pupil_data = np.array(pupil_data)
    body_data = np.array(body_data)
    mesoscale_data = np.array(mesoscale_data)
    pupillometry_data = np.array(pupillometry_data)

    _plot_pupil(pupil_data, args.frames_left, args.skip_every,
            args.frames_of_interest)
    _plot_body(body_data, args.frames_left, args.skip_every,
            args.frames_of_interest)
    _plot_mesoscale(
            mesoscale_data, args.frames_left, args.skip_every,
            args.frames_of_interest)
    _plot_pupillometry(
            pupillometry_data, args.frames_left, args.skip_every,
            args.frames_of_interest)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_args = config_to_namespace(args.config)

    main(config_args)
