import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as ani
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
SAVE_DIR = "/Users/christian/Documents/summer2023/MesoNet/data_events/test/"
# CAPTURE_WINDOW = 0.05  # Amount of time to capture before and after the event.
CAPTURE_WINDOW = 2  # Amount of time to capture before and after the event.
PUPIL_START_FRAME = PUPIL_EVENT_FRAMES[0] - MESOSCALE_START_EVENT_FRAME + 1


i = 0

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    pupil_video_fps = cv2.VideoCapture(PUIPIL_FILENAME).get(cv2.CAP_PROP_FPS)
    mesoscale_fps = 30.0  # TODO
    image_series = ImageSeriesCreator.create(MESOSCALE_PREPROCESSED_FILENAME,
                                             128,
                                             128,
                                             "all")

    N = 20
    images = [[] for _ in range(N)]
    start_time = PUPIL_START_FRAME / pupil_video_fps
    for pupil_event_frame in PUPIL_EVENT_FRAMES:
        time = pupil_event_frame / pupil_video_fps
        time = time + 0.04  # 40 milliseconds.
        time = time - start_time
        mesoscale_event_frame = int(time * mesoscale_fps)
        for i in range(N):
            images[i].append(image_series.get_frame(mesoscale_event_frame - 1 + i))

    average_image = np.mean(np.array(images), axis=0)

    plt.rcParams.update({'font.size': 5})
    _, axes = plt.subplots(1, N)
    for i, image in enumerate(images):
        average_image = np.mean(np.array(image), axis=0)
        axes[i].set_title(f"+{(i + 1) / mesoscale_fps:.2f} s")
        axes[i].axis('off')
        axes[i].imshow(average_image)
    plt.show()


    # n_frames = image_series.image_array.shape[0]
    # n_window_frames = np.floor(CAPTURE_WINDOW * pupil_video_fps)

    # print(image_series.image_array.shape)
    # print(n_frames)

    # for event_frame in PUPIL_EVENT_FRAMES:
    #     event_frame_index = event_frame - 1
    #     earliest_frame_index = int(max(event_frame_index - n_window_frames, 0))
    #     latest_frame_index = int(min(event_frame_index + n_window_frames, n_frames))

    #     print(earliest_frame_index)
    #     print(latest_frame_index)

    #     video_array = image_series.image_array[earliest_frame_index:latest_frame_index]
    #     video_filename = os.path.join(SAVE_DIR, f"event_{event_frame}.avi")
    #     print(image_series.image_array.dtype)

    #     # figure = plt.figure()
    #     # global i
    #     # i = earliest_frame
    #     # image = plt.imshow(image_series.image_array[0], animated=True)
    #     # def update(*args):
    #     #     global i
    #     #     i += 1
    #     #     if i >= latest_frame:
    #     #         return
    #     #     image.set_array(image_series.image_array[i])
    #     #     return image
    #     # animation = ani.FuncAnimation(figure,
    #     #                               update,
    #     #                               frames=latest_frame - earliest_frame,
    #     #                               blit=True)
    #     # plt.show()

    #     video_writer = cv2.VideoWriter(video_filename,
    #                                    cv2.VideoWriter_fourcc(*"MJPG"),
    #                                    pupil_video_fps,
    #                                    (128, 128),
    #                                    False)

    #     print(np.max(image_series.image_array[0]))
    #     print(np.min(image_series.image_array[0]))

    #     for i in range(earliest_frame_index, latest_frame_index):
    #         frame = image_series.get_frame(i)
    #         plt.clf()
    #         plt.imshow(frame)
    #         plt.title(f"Frame {i}")
    #         plt.pause(0.03)

    #         # frame = frame.astype(np.float64) / np.max(frame)
    #         # frame = (2 ** 8 - 1) * frame
    #         # frame = frame.astype(np.uint8)
    #         # video_writer.write(frame)

    #         # plt.imshow(frame)
    #         # plt.show()
    #         # frame = cv2.normalize(frame, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #         # frame = cv2.merge([frame, frame, frame])
    #         # print(frame)
    #         # cv2.imshow('frame', frame)
    #         # video_writer.write(frame)
    #         # plt.imshow(frame)
    #         # plt.show()
    #         # cv2.imshow("Window", frame)

    #     video_writer.release()


if __name__ == "__main__":
    main()
