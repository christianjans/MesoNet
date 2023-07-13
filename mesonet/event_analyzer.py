import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from helpers.image_series import ImageSeriesCreator

PUPIL_EVENT_FRAMES = [
    313,
    3916,
    7520,
    11123,
    14727,
    18330,
    21912,
    25515,
    29119,
    32722,
]
MESOSCALE_START_EVENT_FRAME = 88

PUIPIL_FILENAME = "/Users/christian/Downloads/fc2_save_2023-02-09-135224-0000.avi"
MESOSCALE_PREPROCESSED_FILENAME = "/Users/christian/Documents/summer2023/matlab/my_data/flash1/02_awake_8x8_30hz_36500fr_FR30Hz_BPF1-5Hz_GSR_DFF0-G4-fr1-36480_FR30Hz_BPF1-5Hz_GSR_DFF0-G4-fr1-36480.raw"
PUPILLOMETRY_CSV_FILE = "/Users/christian/Documents/summer2023/pupillometry_matlab/example_flash1/fc2_save_2023-02-09-135224-0000Pupil Radii.csv"
SAVE_DIR = "/Users/christian/Documents/summer2023/MesoNet/data_events/test/"

P_FPS = cv2.VideoCapture(PUIPIL_FILENAME).get(cv2.CAP_PROP_FPS)
M_FPS = float(os.path.basename(MESOSCALE_PREPROCESSED_FILENAME).split("_")[5][2:-2])

M_F_BEFORE = -15
M_F_AFTER = 15
M_F_TOTAL = M_F_AFTER - M_F_BEFORE


def time_from_pframe(pframe: int) -> float:
    return pframe * P_FPS


def pframe_to_mframe(pframe: int) -> int:
    abs_time = time_from_pframe(pframe)
    m_start_time = time_from_pframe(
            PUPIL_EVENT_FRAMES[0] - MESOSCALE_START_EVENT_FRAME)
    frame = round((abs_time - m_start_time) / M_FPS)

    if frame <= 0:
        raise ValueError(f"Invalid frame number: {frame}")
    return frame


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    image_series = ImageSeriesCreator.create(MESOSCALE_PREPROCESSED_FILENAME,
                                             128,
                                             128,
                                             "all")
    pdata: np.ndarray = np.genfromtxt(PUPILLOMETRY_CSV_FILE, delimiter=";")
    pdata_interval = int(pdata[1][0] - pdata[0][0])
    pdata = {int(frame): pupil_size for frame, pupil_size in pdata}

    pupil_sizes: List[List[float]] = []
    images: List[List[np.ndarray]] = []
    # images = [[] for _ in range(M_F_TOTAL)]
    for i, pupil_event_frame in enumerate(PUPIL_EVENT_FRAMES):
        # Collect pupil size data.
        pupil_sizes.append([])
        for j in range(M_F_BEFORE * pdata_interval, M_F_AFTER * pdata_interval):
            if pupil_event_frame - 1 + j in pdata:
                pupil_sizes[i].append(pdata[pupil_event_frame - 1 + j])

        # Collect mesoscale brain data.
        images.append([])
        mframe = pframe_to_mframe(pupil_event_frame)
        for j in range(M_F_BEFORE, M_F_AFTER):
            images[i].append(image_series.get_frame(mframe - 1 + j))
            # images[j].append(image_series.get_frame(mframe - 1 + j))

    # Take the average over the collected data.
    average_pupil_sizes = np.mean(np.array(pupil_sizes), axis=0)
    average_images = np.mean(np.array(images), axis=0)

    # Plot the data.
    plt.rcParams.update({'font.size': 5})
    pticks = [f"{pdata_interval * i / P_FPS:.3f}"
              for i in range(M_F_BEFORE, M_F_AFTER)]
    mticks = [f"{i / M_FPS:.3f}" for i in range(M_F_BEFORE, M_F_AFTER)]

    pfigure = plt.figure(1)
    paxes = pfigure.gca()
    paxes.set_ylabel("pupil radius (pixels)")
    paxes.set_xlabel("time (s)")
    paxes.set_xticks(range(M_F_TOTAL), pticks)
    paxes.plot(average_pupil_sizes)

    _, maxes = plt.subplots(1, M_F_TOTAL)
    for i, average_image in enumerate(average_images):
        maxes[i].set_title(f"{mticks[i]} s")
        maxes[i].axis('off')
        maxes[i].imshow(average_image)

    plt.show()


if __name__ == "__main__":
    main()
