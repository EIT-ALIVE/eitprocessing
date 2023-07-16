"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to parts of electrical impedance tomographs
as they are read.
"""

import copy
import functools
from dataclasses import dataclass
from dataclasses import field
from enum import auto
from pathlib import Path
from typing import Dict
from typing import List
import numpy as np
from numpy.typing import NDArray
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
    """Enum indicating the vendor (manufacturer) of the EIT device with which the data was
    gathered"""

    DRAEGER = auto()
    TIMPEL = auto()
    SENTEC = auto()
    DRAGER = DRAEGER
    DRÄGER = DRAEGER  # pylint: disable = non-ascii-name


@dataclass(eq=False)
class Sequence:
    """Sequence of timepoints containing EIT and/or waveform data

    A Sequence is meant as a representation of a continuous set of data, either EIT frames,
    waveform data, or both. A Sequence could consist of an entire measurement, a section of a
    measurement, a single breath or even a portion of a breath.

    EIT data is contained within Framesets. A Frameset shares the time axis with a Sequence.

    """

    path: Path | str | List[Path | str] = None
    vendor: Vendor = None
    time: np.ndarray = None
    nframes: int = None
    framerate: int = None
    framesets: Dict[str, Frameset] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list, repr=False)
    timing_errors: List[TimingError] = field(default_factory=list, repr=False)
    phases: List[PhaseIndicator] = field(default_factory=list, repr=False)


    def __post_init__(self):
        self._set_vendor_class()


    def __len__(self) -> int:
        return self.nframes


    def __eq__(self, other) -> bool:
        for attr in ["nframes", "framerate", "framesets", "vendor"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        for attr in ["time", "phases", "events", "timing_errors"]:
            self_attr, other_attr = getattr(self, attr), getattr(other, attr)
            if len(self_attr) != len(other_attr):
                return False
            if not np.all(np.equal(self_attr, other_attr)):
                return False

        return True


    def _set_vendor_class(self):
        if isinstance(self.vendor, str):
            self.vendor = Vendor(self.vendor.lower())

        if self.vendor == Vendor.DRAEGER:
            self.__class__ = DraegerSequence
        elif self.vendor == Vendor.TIMPEL:
            self.__class__ = TimpelSequence
        elif self.vendor is not None:
            raise NotImplementedError(f'vendor {self.vendor} is not implemented')


    @staticmethod
    def check_equivalence(a: "Sequence", b: "Sequence"):
        if (a_ := a.vendor) != (b_ := b.vendor):
            raise TypeError(f"Vendors are not equal: {a_}, {b_}")
        if (a_ := a.framerate) != (b_ := b.framerate):
            raise ValueError(f"Framerates are not equal: {a_}, {b_}")
        if (a_ := a.framesets.keys()) != (b_ := b.framesets.keys()):
            raise AttributeError(
                f"Sequences do not contain the same framesets: {a_}, {b_}"
            )
        return True


    @classmethod
    def merge(cls, a: "Sequence", b: "Sequence") -> "Sequence":
        try:
            Sequence.check_equivalence(a, b)
        except Exception as e:
            raise type(e)(f"Could not merge. {e}")

        path = [a.path, b.path]
        nframes = len(a) + len(b)
        time = np.concatenate((a.time, b.time))
        framesets = {
            name: Frameset.merge(a.framesets[name], b.framesets[name])
            for name in a.framesets
        }

        def merge_attribute(attr: str) -> list:
            a_items = getattr(a, attr)
            b_items = getattr(b.deepcopy(), attr)  # deepcopy avoids overwriting
            for item in b_items:
                item.index += a.nframes
                item.time = time[item.index]
            return a_items + b_items

        return cls(
            path=path,
            vendor=a.vendor,
            time=time,
            nframes=nframes,
            framerate=a.framerate,
            framesets=framesets,
            events=merge_attribute("events"),
            timing_errors=merge_attribute("timing_errors"),
            phases=merge_attribute("phases"),
        )

    @classmethod
    def from_path(
        cls,
        path: Path | str | List[Path | str],
        *args, **kwargs,
    ) -> "Sequence":
        """Load sequence from path(s)

        Args:
            path (Path | str | List[Path | str]): path(s) to data file
            vendor (Vendor | str): vendor indicating the device used
            framerate (int, optional): framerate at which the data was recorded.
                Default for Draeger: 20
                Default for Timpel: 50
            first_frame (int, optional): select the starting frame.
                Defaults to 0.
            n_frames (int, optional): select the maximum number of frames to load.
                The actual number of frames can be lower than this is the total
                file size would be exceeded.

        Raises:
            NotImplementedError: is raised when there is no loading method for
            the given vendor.

        Returns:
            Sequence: a sequence containing the loaded data from all files in path
        """

        if not isinstance(path, list):
            path = [path]

        sequences = []
        for single_path in path:
            Path(single_path).resolve(strict=True)  # checks that file exists
            sequences.append(cls._load_file(single_path, *args, **kwargs))
        return functools.reduce(cls.merge, sequences)

    @classmethod
    def _load_file( #pylint: disable=too-many-arguments
        cls,
        path: Path | str,
        vendor: Vendor | str,
        framerate: int = None,
        first_frame: int = 0,
        nframes: int | None = None,
    ) -> "Sequence":

        if first_frame is None:
            first_frame = 0
        if int(first_frame) != first_frame:
            raise TypeError(f'''`first_frame` must be an int, but was given as
                            {first_frame} (type: {type(first_frame)})''')
        if not first_frame >= 0:
            raise ValueError(f'''`first_frame` can not be negative, but was
                            given as {first_frame}''')
        first_frame = int(first_frame)

        obj = cls(
            path=Path(path),
            vendor=vendor,
            nframes=nframes,
        )
        obj._set_vendor_class()
        if framerate:
            obj.framerate = framerate
        elif obj.vendor == Vendor.DRAEGER:
            obj.framerate = 20
        elif obj.vendor == Vendor.TIMPEL:
            obj.framerate = 50
        else:
            raise NotImplementedError(
                f'''No default `framerate` for {obj.vendor} data is implemented.
                \n`framerate` must be specified when calling `_load_file` for
                this vendor.''')

        # function from child class, which will load and assign
        # nframes, framesets, events, timing_errors, phases
        obj._load_data(first_frame)

        return obj


    def _load_data(self, first_frame: int | None):
        raise NotImplementedError(
            f"Data loading for {self.vendor} is not implemented")


    def select_by_indices(self, indices) -> "Sequence":
        if not isinstance(indices, slice):
            raise NotImplementedError(
                """Slicing only implemented using a slice object"""
            )
        if indices.step not in (None, 1):
            raise NotImplementedError(
                """Skipping intermediate frames while slicing is not
                currently implemented."""
            )
        if indices.start is None:
            indices = slice(0, indices.stop, indices.step)
        if indices.stop is None:
            indices = slice(indices.start, self.nframes, indices.step)

        obj = self.deepcopy()
        obj.time = self.time[indices]
        obj.nframes = len(obj.time)

        obj.framesets = {
            k: v.select_by_indices(indices) for k, v in self.framesets.items()
        }

        r = range(indices.start, indices.stop)
        for attr in ['events', 'timing_errors', 'phases']:
            setattr(
                obj, attr, [x for x in getattr(obj, attr)
                    if x.index in r]
            )
            for x in getattr(obj, attr):
                x.index -= indices.start

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
    vendor: Vendor = Vendor.DRAEGER

    def _load_data(self, first_frame: int):
        FRAME_SIZE_BYTES = 4358

        # Need to load 1 frame before first to check if there is an event
        # marker. Data for initial frame will not be added to self.
        first_load = 0
        if first_frame > 0:
            first_load = first_frame - 1

        file_size = self.path.stat().st_size
        if file_size % FRAME_SIZE_BYTES:
            raise OSError(
                f"""File size {file_size} not divisible by {FRAME_SIZE_BYTES}.\n
                Make sure this is a valid and uncorrupted Draeger data file."""
            )
        total_frames = file_size // FRAME_SIZE_BYTES
        if self.nframes is not None:
            self.nframes = min(total_frames - first_load, self.nframes)
        else:
            self.nframes = total_frames - first_load

        self.time = np.zeros(self.nframes)
        pixel_values = np.zeros((self.nframes, 32, 32))

        with open(self.path, "br") as fh:
            fh.seek(first_load * FRAME_SIZE_BYTES)
            reader = Reader(fh)

            previous_marker = None
            for index in range(self.nframes):
                use_index = index
                if first_frame > 0:
                    use_index = index-1
                previous_marker = self._read_frame(reader, use_index, pixel_values, previous_marker)

        if first_frame > 0:
            self.nframes -= 1
            self.time = self.time[:-1]
            pixel_values = pixel_values[:-1,:,:]

        params = {"framerate": self.framerate}
        self.framesets["raw"] = Frameset(
            name="raw",
            description="raw impedance data",
            params=params,
            pixel_values=pixel_values,
        )


    def _read_frame(
        self, reader: Reader,
        index: int,
        pixel_values: NDArray,
        previous_marker: int | None,
    ) -> None:
        current_time = round(reader.float64() * 24 * 60 * 60, 3)

        _ = reader.float32()
        pixel_values[index, :, :] = self.reshape_frame(reader.float32(length=1024))
        min_max_flag = reader.int32()
        event_marker = reader.int32()
        event_text = reader.string(length=30)
        timing_error = reader.int32()

        # TODO: parse medibus data into waveform data
        medibus_data = reader.float32(  # noqa; variable will be used in future version
            length=52
        )

        if index >= 0:
            # The event marker stays the same until the next event occurs. Therefore, check whether the
            # event marker has changed with respect to the most recent event. If so, create a new event.
            if (previous_marker is not None) and (event_marker > previous_marker):
                self.events.append(Event(index, current_time, event_marker, event_text))
            if timing_error:
                self.timing_errors.append(TimingError(index, current_time, timing_error))
            if min_max_flag == 1:
                self.phases.append(MaxValue(index, current_time))
            elif min_max_flag == -1:
                self.phases.append(MinValue(index, current_time))
            self.time[index] = current_time

        return event_marker

    @staticmethod
    def reshape_frame(frame):
        return np.reshape(frame, (32, 32), "C")


@dataclass(eq=False)
class TimpelSequence(Sequence):
    vendor: Vendor = Vendor.TIMPEL

    def _load_data(self, first_frame: int):
        COLUMN_WIDTH = 1030

        try:
            data = np.loadtxt(
            self.path,
            dtype=float,
            delimiter=",",
            skiprows=first_frame,
            max_rows=self.nframes,
        )
        except UnicodeDecodeError as e:
            raise OSError(
                f"""File {self.path} could not be read as Timpel data.\n
                Make sure this is a valid and uncorrupted Timpel data file.\n
                Original error message: {e}"""
            ) from e

        if data.shape[1] != COLUMN_WIDTH:
            raise OSError(
                f"""Input does not have a width of {COLUMN_WIDTH} columns.\n
                Make sure this is a valid and uncorrupted Timpel data file."""
            )
        self.nframes = data.shape[0]

        # TODO: check whether below issue was only a Drager problem or also
        # applicable to Timpel.
        # The implemented method seems convoluted: it's easier to create an array
        # with nframes and add a time_offset. However, this results in floating
        # point errors, creating issues with comparing times later on.
        self.time = np.arange(self.nframes + first_frame) / self.framerate
        self.time = self.time[first_frame:]


        pixel_data = data[:, :1024]
        pixel_data = np.reshape(pixel_data, newshape=(-1, 32, 32), order="C")
        pixel_data = np.where(pixel_data == -1000, np.nan, pixel_data)

        # extract waveform data
        waveform_data = {
            "airway_pressure": data[:, 1024],
            "flow": data[:, 1025],
            "volume": data[:, 1026],
        }

        # extract breath start, breath end and QRS marks
        for index in np.flatnonzero(data[:, 1027] == 1):
            self.phases.append(MinValue(index, self.time[index]))

        for index in np.flatnonzero(data[:, 1028] == 1):
            self.phases.append(MaxValue(index, self.time[index]))

        for index in np.flatnonzero(data[:, 1029] == 1):
            self.phases.append(QRSMark(index, self.time[index]))

        self.phases.sort(key=lambda x: x.index)

        self.framesets["raw"] = Frameset(
            name="raw",
            description="raw timpel data",
            params={"framerate": self.framerate},
            pixel_values=pixel_data,
            waveform_data=waveform_data,
        )
