from __future__ import annotations
from typing import Iterable, Tuple


class EventFrames:
    def __init__(self, event_frames: Iterable[int], zero_index: bool = True):
        if zero_index:
            self._event_frames = [frame - 1 for frame in event_frames]
        else:
            self._event_frames = [frame for frame in event_frames]
        self._assert_event_frames()

    def __len__(self) -> int:
        return len(self._event_frames)

    @property
    def min(self) -> int:
        return self._event_frames[0]
    
    @property
    def max(self) -> int:
        return self._event_frames[-1]

    def event_frame(self, event_index: int) -> int:
        return self._event_frames[event_index]

    def event_index(self, frame: int) -> Tuple[int, float]:
        """Get the index of the event that occurs at or before the provided
        frame. Additionally, the fraction of the given frame in between its
        two neighboring events is given.
        """
        if frame < self.event_frame(0) or frame > self.event_frame(-1):
            raise ValueError(f"Cannot get event index for frame {frame}")

        i = 0
        while i < len(self) and frame >= self.event_frame(i):
            i += 1

        event_index = i - 1
        event_frame = self.event_frame(event_index)

        if frame in self._event_frames:
            return event_index, 0.0

        fraction = (frame - event_frame) / (self.event_frame(event_index + 1) - event_frame)
        return event_index, fraction

    def equivalent_frame(self,
                         other_frame: int,
                         other_event_frames: EventFrames):
        assert len(other_event_frames) == len(self)

        event_index, fraction = other_event_frames.event_index(other_frame)

        if event_index == len(self) - 1:
            return self.event_frame(event_index)

        span = self.event_frame(event_index + 1) - self.event_frame(event_index)
        return round(self.event_frame(event_index) + fraction * span)

    def _assert_event_frames(self):
        assert isinstance(self._event_frames, list)
        assert len(self) > 1
        assert all([frame >= 0 for frame in self._event_frames])
        assert all([
            self._event_frames[i] < self._event_frames[i + 1]
            for i in range(0, len(self._event_frames) - 1)
        ])


if __name__ == "__main__":
    # Used for testing.

    l1 = [156, 3751, 7346, 10942, 14537, 18132, 21728, 25323]
    l2 = [272, 3872, 7472, 11072, 14672, 18272, 21872, 25472]
    l3 = [225, 3828, 7432, 11035, 14639, 18242, 21846, 25449]
    e1 = EventFrames(l1, zero_index=True)
    e2 = EventFrames(l2, zero_index=True)
    e3 = EventFrames(l3, zero_index=True)

    for i, f in enumerate(l1):
        assert e2.equivalent_frame(f - 1, e1) == l2[i] - 1
    for i, f in enumerate(l2):
        assert e3.equivalent_frame(f - 1, e2) == l3[i] - 1
    for i, f in enumerate(l3):
        assert e1.equivalent_frame(f - 1, e3) == l1[i] - 1

    previous_frame = l2[0]
    for i in range(l1[0] - 1, l1[1] + 1 - 1):
        equivalent_frame = e2.equivalent_frame(i, e1)
        assert equivalent_frame - previous_frame < 3
        previous_frame = equivalent_frame

    previous_frame = l2[-2]
    for i in range(l1[-2] - 1, l1[-1] + 1 - 1):
        equivalent_frame = e2.equivalent_frame(i, e1)
        assert equivalent_frame - previous_frame < 3
        previous_frame = equivalent_frame

    for i in range(l1[0] - 1, l1[-1] + 1 - 1):
        equivalent_frame = e1.equivalent_frame(i, e1)
        assert i == equivalent_frame
