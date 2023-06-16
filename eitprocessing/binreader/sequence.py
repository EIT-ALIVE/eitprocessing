"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to parts of electrical impedance tomographs 
as they are read.
"""

import copy
import functools
import itertools
from dataclasses import dataclass
from dataclasses import field
from enum import auto
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
import numpy as np
from strenum import LowercaseStrEnum
from .event import Event
from .frameset import Frameset
from .phases import MaxValue
from .phases import MinValue
from .phases import PhaseIndicator
from .phases import QRSMark
from .reader import Reader
from .timing_error import TimingError


class Vendor(LowercaseStrEnum):
    DRAEGER = auto()
    TIMPEL = auto()
    SENTEC = auto()
    DRAGER = DRAEGER
    DRÄGER = DRAEGER


@dataclass(eq=False)
class Sequence:
    path: Path | str = None
    time: np.ndarray = None
    n_frames: int = None
    framerate: int = None
    framesets: Dict[str, Frameset] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list, repr=False)
    timing_errors: List[TimingError] = field(default_factory=list, repr=False)
    phases: List[PhaseIndicator] = field(default_factory=list, repr=False)
    vendor: Vendor = None

    def __len__(self) -> int:
        return self.n_frames

    def __eq__(self, other) -> bool:
        for attr in ["n_frames", "framerate", "framesets", "vendor"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        for attr in ["time", "phases", "events", "timing_errors"]:
            self_attr, other_attr = getattr(self, attr), getattr(other, attr)
            if len(self_attr) != len(other_attr):
                return False
            if not np.all(np.equal(self_attr, other_attr)):
                return False

        return True

    @classmethod
    def merge(cls, a, b) -> "Sequence":  # pylint: disable = too-many-locals
        path = list(itertools.chain([a.path, b.path]))

        if a.vendor != b.vendor:
            raise ValueError("Vendors aren't equal")

        if (a_ := a.framerate) != (b_ := b.framerate):
            raise ValueError(f"Framerates are not equal: {a_}, {b_}")

        # Create time axis
        n_frames = len(a) + len(b)
        time = np.arange(n_frames) / a.framerate + a.time[0]

        # Merge framesets
        if (a_ := a.framesets.keys()) != (b_ := b.framesets.keys()):
            raise AttributeError(
                f"Sequences don't contain the same framesets: {a_}, {b_}"
            )
        framesets = {
            name: Frameset.merge(a.framesets[name], b.framesets[name])
            for name in a.framesets.keys()
        }

        def merge_list_attribute(attr: str) -> list:
            a_items = getattr(a, attr)
            b_items = copy.deepcopy(
                getattr(b, attr)
            )  # make a copy to prevent overwriting b
            for item in b_items:
                item.index += a.n_frames
                item.time = time[item.index]
            return a_items + b_items

        return cls(
            path=path,
            time=time,
            n_frames=n_frames,
            framerate=a.framerate,
            framesets=framesets,
            events=merge_list_attribute("events"),
            timing_errors=merge_list_attribute("timing_errors"),
            phases=merge_list_attribute("phases"),
            vendor=a.vendor,
        )

    @classmethod
    def from_paths(
        cls, paths: List[Path], vendor: Vendor, framerate: int = None
    ) -> "Sequence":
        sequences = (
            cls.from_path(path, framerate=framerate, vendor=vendor) for path in paths
        )
        return functools.reduce(lambda a, b: cls.merge(a, b), sequences)

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        vendor: Vendor,
        framerate: int = None,
        limit_frames: slice | Tuple[int, int] = None,
    ) -> "Sequence":
        """Load sequence from path

        Args:
            path (Path | str): path to data file
            vendor (Vendor): vendor indicating the device used
            framerate (int, optional): framerate at which the data was recorded. Defaults to None.
            limit_frames (slice | Tuple[int, int], optional): limit the range of frames to be loaded. Defaults to None.

        Raises:
            NotImplementedError: is raised when there is no loading method for the given vendor.

        Returns:
            Sequence: a sequence containing the loaded data
        """

        path = Path(path)

        if vendor == Vendor.DRAEGER:
            return DraegerSequence.from_path(path, framerate, limit_frames)

        if vendor == Vendor.TIMPEL:
            return TimpelSequence.from_path(path, framerate, limit_frames)

        raise NotImplementedError(f"cannot load data from vendor {vendor}")

    @staticmethod
    def parse_limit_frames(limit_frames: tuple | slice) -> slice:
        if not limit_frames:
            return None
        
        if isinstance(limit_frames, tuple):
            if len(limit_frames) != 2:
                raise ValueError(
                    "`limit_frames` should be a tuple (start, stop) or a slice."
                )
            limit_frames = slice(*limit_frames)

        if limit_frames and not isinstance(limit_frames, slice):
            raise TypeError(
                f"`limit_frames` should be a slice or tuple (to be converted to a slice), not {type(limit_frames)!r}"
            )

        if limit_frames.step != 1 and limit_frames.step is not None:
            raise NotImplementedError("Can't skip frames when loading data.")

        if limit_frames.start is None and limit_frames.stop is None:
            return None

        if limit_frames.start is None:
            limit_frames = slice(0, limit_frames.stop)

        return limit_frames

    def select_by_indices(self, indices) -> "Sequence":
        obj = self.deepcopy()

        obj.framesets = {
            k: v.select_by_indices(indices) for k, v in self.framesets.items()
        }
        obj.time = self.time[indices]
        obj.n_frames = len(obj.time)

        if isinstance(indices, slice):
            if indices.start is None:
                indices = slice(0, indices.stop, indices.step)
            first = indices.start
        else:
            first = indices[0]

        def filter_by_index(list_):
            def helper(item):
                if isinstance(indices, slice):
                    if indices.step not in (None, 1):
                        raise NotImplementedError(
                            "Can't skip intermediate frames while slicing"
                        )
                    return item.index >= indices.start and (indices.stop is None or item.index < indices.stop)
                return item.index in indices

            new_list = list(filter(helper, list_))
            for item in new_list:
                item.index = item.index - first
                item.time = obj.time[item.index]

            return new_list

        obj.events = filter_by_index(obj.events)
        obj.timing_errors = filter_by_index(obj.timing_errors)
        obj.phases = filter_by_index(obj.phases)

        return obj

    def select_by_time(self, start=None, end=None, end_inclusive=False) -> "Sequence":
        if not any((start, end)):
            raise ValueError("Pass either start or end")
        start_index = np.nonzero(self.time >= start)[0][0]
        if end_inclusive:
            end_index = np.nonzero(self.time <= end)[0][-1]
        else:
            end_index = np.nonzero(self.time < end)[0][-1]

        return self.select_by_indices(slice(start_index, end_index))

    __getitem__ = select_by_indices
    deepcopy = copy.deepcopy

@dataclass(eq=False)
class DraegerSequence(Sequence):
    framerate: int = 20
    vendor: Vendor = Vendor.DRAEGER

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        framerate: int = None,
        limit_frames: slice | Tuple[int, int] = None,
    ) -> "DraegerSequence":
        obj = cls(path=Path(path))

        if framerate:
            obj.framerate = framerate

        limit_frames = obj.parse_limit_frames(limit_frames)

        skip_frames = 0
        if limit_frames:
            skip_frames = limit_frames.start

        obj.read(limit_frames=limit_frames)
        obj.time = np.arange(len(obj) + skip_frames) / obj.framerate
        obj.time = obj.time[skip_frames:]

        return obj

    def read(self, limit_frames: slice = None, framerate: int = 20) -> None:
        FRAME_SIZE_BYTES = 4358

        file_size = self.path.stat().st_size
        if file_size % FRAME_SIZE_BYTES:
            raise ValueError(
                f"File size {file_size} not divisible by {FRAME_SIZE_BYTES}"
            )

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
                self.read_frame_draeger(reader, index, pixel_values)

        params = {"framerate": framerate}
        self.framesets["raw"] = Frameset(
            name="raw",
            description="raw impedance data",
            params=params,
            pixel_values=pixel_values,
        )

    def read_frame(self, reader, index, pixel_values) -> None:
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

        # TODO: parse medibus data into waveform data
        medibus_data = reader.float32(
            length=52
        )  # noqa; code crashes if line is removed

        # The event marker stays the same until the next event occurs. Therefore, check whether the
        # event marker has changed with respect to the most recent event. If so, create a new event.
        if self.events:
            previous_event = self.events[-1]
        else:
            previous_event = None

        if event_marker and (
            previous_event is None or event_marker > previous_event.marker
        ):
            self.events.append(Event(index, event_marker, event_text))

        if timing_error:
            self.timing_errors.append(TimingError(index, time, timing_error))

        if min_max_flag == 1:
            self.phases.append(MaxValue(index, time))
        elif min_max_flag == -1:
            self.phases.append(MinValue(index, time))

@dataclass(eq=False)
class TimpelSequence(Sequence):
    framerate: int = 50
    vendor: Vendor = Vendor.TIMPEL

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        framerate: int = None,
        limit_frames: slice | Tuple[int, int] = None,
    ) -> "TimpelSequence":
        obj = cls(path=Path(path))

        if framerate:
            obj.framerate = framerate

        limit_frames = obj.parse_limit_frames(limit_frames)

        skiprows, max_rows = 0, None
        if limit_frames:
            skiprows = limit_frames.start
            if limit_frames.stop:
                max_rows = limit_frames.stop - limit_frames.start
            

        data = np.loadtxt(
            path, dtype=float, delimiter=",", skiprows=skiprows, max_rows=max_rows
        )

        obj.n_frames = data.shape[0]
        if max_rows and max_rows != obj.n_frames:
            raise ValueError(
                "Fewer rows were loaded than indicated with `limit_frames`"
            )

        # Below method seems convoluted: it's easier to create an array with n_frames and add a
        # time_offset. However, this results in floating points errors, creating issues with
        # comparing times later on.
        obj.time = np.arange(obj.n_frames + skiprows) / obj.framerate
        obj.time = obj.time[skiprows:]

        if data.shape[1] != 1030:
            raise ValueError("CSV file does not contain 1030 columns")

        pixel_data = data[:, :1024]
        pixel_data = np.reshape(pixel_data, newshape=(-1, 32, 32), order="C")
        pixel_data = np.where(pixel_data == -1000, np.nan, pixel_data)

        # extract waveform data
        waveform_data = dict(
            airway_pressure=data[:, 1024], flow=data[:, 1025], volume=data[:, 1026]
        )

        # extract breath start, breath end and QRS marks
        for index in np.flatnonzero(data[:, 1027] == 1):
            obj.phases.append(MinValue(index, obj.time[index]))

        for index in np.flatnonzero(data[:, 1028] == 1):
            obj.phases.append(MaxValue(index, obj.time[index]))

        for index in np.flatnonzero(data[:, 1029] == 1):
            obj.phases.append(QRSMark(index, obj.time[index]))

        obj.phases.sort(key=lambda x: x.index)

        obj.framesets["raw"] = Frameset(
            name="raw",
            description="raw timpel data",
            params=dict(framerate=obj.framerate),
            pixel_values=pixel_data,
            waveform_data=waveform_data,
        )

        return obj
