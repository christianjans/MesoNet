import cv2
import matplotlib.pyplot as plt
import numpy as np


# video_capture = cv2.VideoCapture("/Users/christian/Downloads/fc2_save_2023-02-09-135224-0000.avi")

# success, image = video_capture.read()
# n_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

# count = 0

# PUPIL_EVENT_FRAMES = [
#     313,
#     3916,
#     7520,
#     # 11123,
#     # 14727,
#     # 18330,
#     # 21912,
#     # 25515,
#     # 29119,
# ]

# for frame_number in PUPIL_EVENT_FRAMES:
#     video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 2)
#     success, frame = video_capture.read()
#     plt.imshow(frame)
#     plt.show()
#     plt.clf()

#     video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
#     success, frame = video_capture.read()
#     cv2.imshow("Video", frame)
#     plt.imshow(frame)
#     plt.show()
#     plt.clf()


class VideoSeries:
    def __init__(self, filename: str):
        self._video_capture = cv2.VideoCapture(filename)
        self._n_frames = self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0 or frame_index >= self._n_frames:
            raise ValueError(f"Frame index {frame_index} is out of range")

        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = self._video_capture.read()

        if not success:
            raise RuntimeError(f"Unable to load frame index {frame_index}")
        return frame
