import argparse
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from mesonet.chan_lab.helpers.image_series import ImageSeriesCreator
from mesonet.chan_lab.helpers.utils import config_to_namespace


def time_from_pframe(pframe: int, p_fps: float) -> float:
    return pframe * p_fps


def pframe_to_mframe(pframe: int,
                     p_fps: float,
                     m_fps: float,
                     pframe_reference: int,
                     mframe_reference: int) -> int:
    abs_time = time_from_pframe(pframe, p_fps)
    m_start_time = time_from_pframe(pframe_reference - mframe_reference, p_fps)
    frame = round((abs_time - m_start_time) / m_fps)

    if frame <= 0:
        raise ValueError(f"Invalid frame number: {frame}")
    return frame


def main(args: argparse.Namespace):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    image_series = ImageSeriesCreator.create_cached_image_series(
            args.mesoscale_file, 128, 128, "all")
    video_series = ImageSeriesCreator.create_uncached_image_series(
            args.pupil_file)
    pdata: np.ndarray = np.genfromtxt(args.pupillometry_file, delimiter=";")
    pdata_interval = int(pdata[1][0] - pdata[0][0])
    pdata = {int(frame): pupil_size for frame, pupil_size in pdata}
    p_fps = cv2.VideoCapture(args.pupil_file).get(cv2.CAP_PROP_FPS)
    m_fps = float(os.path.basename(args.mesoscale_file).split("_")[5][2:-2])
    frames_total = args.frames_after - args.frames_before

    pupil_sizes: List[List[float]] = []
    video_frames: List[List[np.ndarray]] = []
    images: List[List[np.ndarray]] = []
    for i, pupil_event_frame in enumerate(args.pupil_event_frames):
        # Collect pupil size data.
        pupil_sizes.append([])
        for j in range(args.frames_before * pdata_interval,
                       args.frames_after * pdata_interval):
            if pupil_event_frame - 1 + j in pdata:
                pupil_sizes[i].append(pdata[pupil_event_frame - 1 + j])

        # Collect video data.
        video_frames.append([])
        for j in range(args.frames_before, args.frames_after):
            video_frame = video_series.get_frame(pupil_event_frame - 1 + j)
            video_frame = np.mean(video_frame, axis=-1)
            video_frame = video_frame.astype(np.uint8)
            video_frame = 255 - video_frame
            video_frames[i].append(video_frame)

        # Collect mesoscale brain data.
        images.append([])
        mframe = pframe_to_mframe(pupil_event_frame,
                                  p_fps,
                                  m_fps,
                                  args.pupil_event_frames[0],
                                  args.mesoscale_event_start_frame)
        for j in range(args.frames_before, args.frames_after):
            images[i].append(image_series.get_frame(mframe - 1 + j))

    # Take the average over the collected data.
    average_pupil_sizes = np.mean(np.array(pupil_sizes), axis=0)
    std_pupil_sizes = np.std(np.array(pupil_sizes), axis=0)
    average_video_frames = np.mean(np.array(video_frames), axis=0)
    average_images = np.mean(np.array(images), axis=0)

    # Plot the data.
    plt.rcParams.update({'font.size': 5})
    pticks = [f"{pdata_interval * i / p_fps:.3f}"
              for i in range(args.frames_before, args.frames_after)]
    mticks = [f"{i / m_fps:.3f}"
              for i in range(args.frames_before, args.frames_after)]

    pfigure = plt.figure(1)
    paxes = pfigure.gca()
    paxes.set_ylabel("average pupil radius (pixels)")
    paxes.set_xlabel("time (s)")
    paxes.set_xticks(range(frames_total), pticks)
    paxes.plot(average_pupil_sizes)
    paxes.fill_between(range(frames_total),
                       average_pupil_sizes - std_pupil_sizes,
                       average_pupil_sizes + std_pupil_sizes,
                       alpha=0.5)

    assert frames_total % args.plot_rows == 0

    mfigure, maxes = plt.subplots(args.plot_rows,
                                  frames_total // args.plot_rows,
                                  dpi=200)
    for i, average_image in enumerate(average_images):
        row = i // (frames_total // args.plot_rows)
        column = i % (frames_total // args.plot_rows)
        maxes[row, column].set_title(f"{mticks[i]}s")
        maxes[row, column].axis('off')
        maxes[row, column].imshow(average_image, vmin=-1.0, vmax=1.0)
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cmap = mpl.cm.viridis
    mfigure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    mfigure.tight_layout()

    vfigure, vaxes = plt.subplots(args.plot_rows,
                                  frames_total // args.plot_rows,
                                  dpi=200,
                                  figsize=(8, 8))
    for i, average_video_frame in enumerate(average_video_frames):
        row = i // (frames_total // args.plot_rows)
        column = i % (frames_total // args.plot_rows)
        vaxes[row, column].set_title(f"{mticks[i]}s")
        vaxes[row, column].axis('off')
        vaxes[row, column].imshow(average_video_frame, vmin=0, vmax=255, cmap="binary")
    vfigure.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_args = config_to_namespace(args.config)

    main(config_args)
