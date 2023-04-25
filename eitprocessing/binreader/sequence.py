"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to parts of electrical impedance tomographs 
as they are read.
"""

import copy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
import numpy as np
from .event import Event
from .frameset import Frameset
from .phases import MaxValue
from .phases import MinValue
from .phases import PhaseIndicator
from .reader import Reader
from .timing_error import TimingError


@dataclass
class Sequence:
    path: Path
    time: np.ndarray = None
    n_frames: int = None
    framerate: int = 20
    framesets: Dict[str, Frameset] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list, repr=False)
    timing_errors: List[TimingError] = field(default_factory=list, repr=False)
    phases: List[PhaseIndicator] = field(default_factory=list, repr=False)

    def __len__(self):
        return self.n_frames

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        framerate: int = None,
        limit_frames: slice | Tuple[int, int] = None,
    ):
        obj = cls(path=Path(path))

        if framerate:
            obj.framerate = framerate

        if isinstance(limit_frames, tuple):
            limit_frames = slice(*limit_frames)

        if limit_frames:
            time_offset = limit_frames.start / framerate
        else:
            time_offset = 0

        obj.read(limit_frames=limit_frames)
        obj.time = np.arange(len(obj)) / framerate + time_offset

        return obj

    def select_by_indices(self, indices):
        obj = self.deepcopy()

        obj.framesets = {k: v.select_by_indices(indices) for k, v in self.framesets.items()}
        obj.time = self.time[indices]
        obj.n_frames = len(obj.time)

        def filter_by_index(list_):
            if isinstance(indices, slice):
                first = indices.start
            else:
                first = indices[0]

            def helper(item):
                if isinstance(indices, slice):
                    if indices.step not in (None, 1):
                        raise NotImplementedError("Can't skip intermediate frames while slicing")
                    return item.index >= indices.start and item.index < indices.stop
                else:
                    return item.index in indices

            new_list = list(filter(helper, list_))
            for item in new_list:
                item.index = item.index - first

            return new_list

        obj.events = filter_by_index(obj.events)
        obj.timing_errors = filter_by_index(obj.timing_errors)
        obj.phases = filter_by_index(obj.phases)

        return obj

    __getitem__ = select_by_indices

    def select_by_time(self, start=None, end=None, end_inclusive=False):
        if not any((start, end)):
            raise ValueError("Pass either start or end")
        start_index = np.nonzero(self.time >= start)[0][0]
        if end_inclusive:
            end_index = np.nonzero(self.time <= end)[0][-1]
        else:
            end_index = np.nonzero(self.time < end)[0][-1]

        return self.select_by_indices(slice(start_index, end_index))

    def read(self, limit_frames: slice = None, framerate: int = 20):
        FRAME_SIZE_BYTES = 4358

        file_size = self.path.stat().st_size
        if file_size % FRAME_SIZE_BYTES:
            raise ValueError(f"File size {file_size} not divisible by {FRAME_SIZE_BYTES}")

        max_n_frames = file_size // FRAME_SIZE_BYTES

        if limit_frames:
            if limit_frames.step not in (1, None):
                raise NotImplementedError("Can't skip frames")

            if limit_frames.stop and limit_frames.stop < max_n_frames:
                stop = limit_frames.stop
            else:
                stop = max_n_frames

            start = limit_frames.start or 0
            self.n_frames = stop - start
        else:
            self.n_frames = max_n_frames

        pixel_values = np.empty((self.n_frames, 32, 32))

        with open(self.path, "br") as fh:
            if limit_frames:
                fh.seek(limit_frames.start * FRAME_SIZE_BYTES)

            reader = Reader(fh)
            for index in range(self.n_frames):
                self.read_frame(reader, index, pixel_values)

        params = {"framerate": framerate}
        self.framesets["raw"] = Frameset(
            name="raw", description="raw impedance data", params=params, pixel_values=pixel_values
        )

    def read_frame(self, reader, index, pixel_values):
        def reshape_frame(frame):
            return np.reshape(frame, (32, 32), "C")

        timestamp = reader.float64()
        time = timestamp * 24 * 60 * 60

        _ = reader.float32()
        pixel_values[index, :, :] = reshape_frame(reader.float32(length=1024))
        min_max_flag = reader.int32()
        event_marker = reader.int32()
        event_text = reader.string(length=30)
        timing_error = reader.int32()
        medibus_data = reader.float32(length=52)

        if len(self.events):
            previous_event = self.events[-1]
        else:
            previous_event = None

        if event_marker and (previous_event is None or event_marker > previous_event.marker):
            self.events.append(Event(index, event_marker, event_text))

        if timing_error:
            self.timing_errors.append(TimingError(index, time, timing_error))

        if min_max_flag == 1:
            self.phases.append(MaxValue(index, time))
        elif min_max_flag == -1:
            self.phases.append(MinValue(index, time))

    deepcopy = copy.deepcopy
