"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to parts of electrical impedance tomographs
as they are read.
"""
from __future__ import annotations
import bisect
import copy
import functools
import warnings
from dataclasses import dataclass
from dataclasses import field
from enum import auto
from pathlib import Path
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
    """Enum indicating the vendor (manufacturer) of the EIT device with which the data was gathered."""

    DRAEGER = auto()
    TIMPEL = auto()
    SENTEC = auto()
    DRAGER = DRAEGER
    DRÄGER = DRAEGER  # pylint: disable = non-ascii-name


@dataclass(eq=False)
class Sequence:
    """Sequence of timepoints containing EIT and/or waveform data.

    A Sequence is a representation of a continuous set of data points, either EIT frames,
    waveform data, or both. A Sequence can consist of an entire measurement, a section of a
    measurement, a single breath, or even a portion of a breath.
    A sequence can be split up into separate sections of a measurement or multiple (similar)
    Sequence objects can be merged together to form a single Sequence.

    EIT data is contained within Framesets. A Frameset shares the time axis with a Sequence.

    Args:
        path (Path | str | list[Path | str]): path(s) to data file.
        vendor (Vendor | str): vendor indicating the device used.
        label (str): description of object for human interpretation.
            Defaults to "Sequence_<unique_id>".
        time (NDArray[float]): list of time label for each data point (can be
            true time or relative time)
        max_frames (int): number of frames in sequence
        framerate (int, optional): framerate at which the data was recorded.
            Defaults to 20 if vendor == DRAEGER
            Defaults to 50 if vendor == TIMPEL
        framesets (dict[str, Frameset]): dictionary of framesets
        events (list[Event]): list of Event objects in data
        timing_errors (list[TimingError]): list of TimingError objects in data
        phases (list[PhaseIndicator]): list of PhaseIndicator objects in data
    """

    path: Path | str | list[Path | str] | None = None
    vendor: Vendor | str | None = None
    label: str | None = None
    time: NDArray | None = None
    nframes: int | None = None
    framerate: int | None = None
    framesets: dict[str, Frameset] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list, repr=False)
    timing_errors: list[TimingError] = field(default_factory=list, repr=False)
    phases: list[PhaseIndicator] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.label is None:
            self.label = f"Sequence_{id(self)}"

        self._set_vendor_class()

    def __len__(self) -> int:
        return self.nframes

    def __eq__(self, other) -> bool:
        try:
            self.check_equivalence(self, other)
        except (TypeError, ValueError, AttributeError):
            return False

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
        """Re-assign Sequence class to child class for selected Vendor.

        Raises:
            NotImplementedError: if the child class for the selected vendor
                has not yet been implemented.
            TypeError: if the `vendor` argument does not correspond to the expected
                vendor for `type(self)`.
        """

        if isinstance(self.vendor, str):
            self.vendor = Vendor(self.vendor.lower())

        if (
            isinstance(self, Sequence)
            and self.__class__ is not Sequence
            and self.__class__.vendor != self.vendor
        ):
            raise TypeError(
                f"`vendor` for {type(self)} cannot be set as {self.vendor}."
            )

        # Note that this way of re-assigning classes is considered to be a bad practice
        # (https://tinyurl.com/2x2cea6h), but the objections raised don't seem to be prohibtive.
        if self.vendor == Vendor.DRAEGER:
            self.__class__ = DraegerSequence
        elif self.vendor == Vendor.TIMPEL:
            self.__class__ = TimpelSequence
        elif self.vendor is not None:
            raise NotImplementedError(f"vendor {self.vendor} is not implemented")

    @staticmethod
    def check_equivalence(a: Sequence, b: Sequence):
        """Checks whether content of two Sequence objects is equivalent."""

        if (a_ := a.vendor) != (b_ := b.vendor):
            raise TypeError(f"Vendors are not equal: {a_}, {b_}")
        if (a_ := a.framerate) != (b_ := b.framerate):
            raise ValueError(f"Framerates are not equal: {a_}, {b_}")
        if (a_ := a.framesets.keys()) != (b_ := b.framesets.keys()):
            raise AttributeError(
                f"Sequences do not contain the same framesets: {a_}, {b_}"
            )
        return True

    def __add__(self, other: Sequence) -> Sequence:
        return self.merge(self, other)

    @classmethod
    def merge(
        cls,
        a: Sequence,
        b: Sequence,
        label: str | None = None,
    ) -> Sequence:
        """Create a merge of two Sequence objects."""

        try:
            Sequence.check_equivalence(a, b)
        except Exception as e:
            raise type(e)(f"Sequences could not be merged: {e}") from e

        a_path = a.path if isinstance(a.path, list) else [a.path]
        b_path = b.path if isinstance(b.path, list) else [b.path]
        path = a_path + b_path
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

        label = f"Merge of <{a.label}> and <{b.label}>" if label is None else label

        return cls(
            path=path,
            vendor=a.vendor,
            label=label,
            time=time,
            nframes=nframes,
            framerate=a.framerate,
            framesets=framesets,
            events=merge_attribute("events"),
            timing_errors=merge_attribute("timing_errors"),
            phases=merge_attribute("phases"),
        )

    @classmethod
    def from_path(  # pylint: disable=too-many-arguments, unused-argument
        cls,
        path: Path | str | list[Path | str],
        vendor: Vendor | str | None = None,
        label: str | None = None,
        framerate: int | None = None,
        first_frame: int = 0,
        max_frames: int | None = None,
    ) -> Sequence:
        """Load sequence from path(s)

        Args:
            path (Path | str | list[Path | str]): path(s) to data file.
            vendor (Vendor | str): vendor indicating the device used.
            label (str): description of object for human interpretation.
                Defaults to "Sequence_<unique_id>".
            framerate (int, optional): framerate at which the data was recorded.
                Default for Draeger: 20
                Default for Timpel: 50
            first_frame (int, optional): index of first time point of sequence
                (i.e. NOT the timestamp).
                Defaults to 0.
            max_frames (int, optional): maximum number of frames to load.
                The actual number of frames can be lower than this if this
                would surpass the final frame.

        Raises:
            NotImplementedError: is raised when there is no loading method for
            the given vendor.

        Returns:
            Sequence: a sequence containing the loaded data from all files in path
        """

        params = list(locals().values())[1:]  # list input parameters
        sequences = []

        if not isinstance(path, list):
            path = [path]
        for single_path in path:
            Path(single_path).resolve(strict=True)  # checks that file exists
            params[0] = single_path
            sequences.append(cls._load_file(*params))
        return functools.reduce(cls.merge, sequences)

    @classmethod
    def _load_file(  # pylint: disable=too-many-arguments
        cls,
        path: Path | str,
        vendor: Vendor | str,
        label: str | None = None,
        framerate: int | None = None,
        first_frame: int = 0,
        max_frames: int | None = None,
    ) -> Sequence:
        """Method used by `from_path` that initiates the object and calls
        child method for loading the data.

        See `from_path` method for arguments."""

        if first_frame is None:
            first_frame = 0
        if int(first_frame) != first_frame:
            raise TypeError(
                f"`first_frame` must be an int, but was given as"
                f" {first_frame} (type: {type(first_frame)})"
            )
        if not first_frame >= 0:
            raise ValueError(
                f"`first_frame` can not be negative, but was given as {first_frame}"
            )
        first_frame = int(first_frame)

        obj = cls(
            path=Path(path),
            vendor=vendor,
            nframes=max_frames,
            label=label,
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
                f"No default `framerate` for {obj.vendor} data is implemented."
                "\n`framerate` must be specified when calling `_load_file` for"
                "this vendor."
            )

        # function from child class, which will load and assign
        # time, nframes, framesets, events, timing_errors, phases
        obj._load_data(first_frame)

        return obj

    def _load_data(self, first_frame: int | None):
        """Needs to be implemented in child class."""
        raise NotImplementedError(f"Data loading for {self.vendor} is not implemented")

    def select_by_index(
        self,
        indices: slice,
        label: str | None = None,
    ):
        if not isinstance(indices, slice):
            raise NotImplementedError("Slicing only implemented using a slice object")
        if indices.step not in (None, 1):
            raise NotImplementedError(
                "Skipping intermediate frames while slicing is not implemented."
            )
        if indices.start is None:
            indices = slice(0, indices.stop, indices.step)
        if indices.stop is None:
            indices = slice(indices.start, self.nframes, indices.step)

        obj = self.deepcopy()
        obj.time = self.time[indices]
        obj.nframes = len(obj.time)
        obj.framesets = {k: v[indices] for k, v in self.framesets.items()}
        obj.label = (
            f"Slice ({indices.start}-{indices.stop}) of <{self.label}>"
            if label is None
            else label
        )

        range_ = range(indices.start, indices.stop)
        for attr in ["events", "timing_errors", "phases"]:
            setattr(obj, attr, [x for x in getattr(obj, attr) if x.index in range_])
            for x in getattr(obj, attr):
                x.index -= indices.start

        return obj

    def __getitem__(self, indices: slice):
        return self.select_by_index(indices)

    def select_by_time(  # pylint: disable=too-many-arguments
        self,
        start: float | int | None = None,
        end: float | int | None = None,
        start_inclusive: bool = True,
        end_inclusive: bool = False,
        label: str | None = None,
    ) -> Sequence:
        """Select subset of sequence by the `Sequence.time` information (i.e.
        based on the time stamp).

        Args:
            start (float | int | None, optional): starting time point.
                Defaults to None.
            end (float | int | None, optional): ending time point.
                Defaults to None.
            start_inclusive (bool, optional): include starting timepoint if
                `start` is present in `Sequence.time`.
                Defaults to True.
            end_inclusive (bool, optional): include ending timepoint if
                `end` is present in `Sequence.time`.
                Defaults to False.

        Raises:
            ValueError: if the Sequence.time is not sorted

        Returns:
            Sequence: a slice of `self` based on time information given.
        """

        if not any((start, end)):
            warnings.warn("No starting or end timepoint was selected.")
            return self
        if not np.all(np.sort(self.time) == self.time):
            raise ValueError(
                f"Time stamps for {self} are not sorted and therefor data"
                "cannot be selected by time."
            )

        if start is None:
            start_index = 0
        elif start_inclusive:
            start_index = bisect.bisect_left(self.time, start)
        else:
            start_index = bisect.bisect_right(self.time, start)

        if end is None:
            end_index = len(self)
        elif end_inclusive:
            end_index = bisect.bisect_right(self.time, end) - 1
        else:
            end_index = bisect.bisect_left(self.time, end) - 1

        return self.select_by_index(slice(start_index, end_index), label=label)

    def deepcopy(
        self,
        label: str | None = None,
        relabel: bool | None = True,
    ) -> Sequence:
        """Create a deep copy of `Sequence` object.

        Args:
            label (str): Create a new `label` for the copy.
                Defaults to None, which will trigger behavior described for relabel (below)
            relabel (bool): If `True` (default), the label of self is re-used for the copy,
                otherwise the following label is assigned f"Deepcopy of {self.label}".
                Note that this setting is ignored if a label is given.

        Returns:
            Sequence: a deep copy of self
        """

        obj = copy.deepcopy(self)
        if label:
            obj.label = label
        elif relabel:
            obj.label = f"Copy of <{self.label}>"
        return obj


@dataclass(eq=False)
class DraegerSequence(Sequence):
    """Sequence object for DRAEGER data."""

    vendor: Vendor = Vendor.DRAEGER

    def _load_data(self, first_frame: int):
        """Load data for DRAEGER files."""
        FRAME_SIZE_BYTES = 4358

        file_size = self.path.stat().st_size
        if file_size % FRAME_SIZE_BYTES:
            raise OSError(
                f"File size {file_size} of file {str(self.path)} not divisible by {FRAME_SIZE_BYTES}.\n"
                f"Make sure this is a valid and uncorrupted Draeger data file."
            )
        total_frames = file_size // FRAME_SIZE_BYTES

        if first_frame > total_frames:
            raise ValueError(
                f"Invalid input: `first_frame` {first_frame} is larger than the "
                f"total number of frames in the file {total_frames}."
            )

        # We need to load 1 frame before first actual frame to check if there
        # is an event marker. Data for the pre-first (dummy) frame will be
        # removed from self at the end of this function.
        first_load = max(0, first_frame - 1)
        loaded_frames = min(
            total_frames - first_load, self.nframes or total_frames - first_load
        )

        self.time = np.zeros(loaded_frames)
        pixel_values = np.zeros((loaded_frames, 32, 32))
        waveform_data = {
            key: np.full((loaded_frames,), np.nan) for key in medibus_field_names
        }

        with open(self.path, "br") as fh:
            fh.seek(first_load * FRAME_SIZE_BYTES)
            reader = Reader(fh)

            previous_marker = None
            for index in range(loaded_frames):
                previous_marker = self._read_frame(
                    reader,
                    index - (first_frame > 0),  # only adjusts for firstframe > 0
                    pixel_values,
                    waveform_data,
                    previous_marker,
                )

        if first_frame > 0:
            # remove data for dummy frame if first frame in file was not loaded
            loaded_frames -= 1
            self.time = self.time[:-1]
            pixel_values = pixel_values[:-1, :, :]
            waveform_data = {key: values[:-1] for key, values in waveform_data.items()}

        if self.nframes != loaded_frames:
            if self.nframes:
                warnings.warn(
                    f"The number of frames requested ({self.nframes}) is larger "
                    f"than the available number ({loaded_frames}) of frames after "
                    f"the first frame selected ({first_frame}, total frames: "
                    f"{total_frames}).\n {loaded_frames} frames have been loaded."
                )
            self.nframes = loaded_frames

        params = {"framerate": self.framerate}
        self.framesets["raw"] = Frameset(
            name="raw",
            description="raw impedance data",
            params=params,
            pixel_values=pixel_values,
            waveform_data=waveform_data,
        )

    def _read_frame(
        self,
        reader: Reader,
        index: int,
        pixel_values: NDArray,
        waveform_data: dict[str, NDArray],
        previous_marker: int | None,
    ):
        """Read frame by frame data from DRAEGER files."""

        current_time = round(reader.float64() * 24 * 60 * 60, 3)

        _ = reader.float32()
        pixel_values[index, :, :] = self.reshape_frame(reader.float32(length=1024))
        min_max_flag = reader.int32()
        event_marker = reader.int32()
        event_text = reader.string(length=30)
        timing_error = reader.int32()

        # TODO (#79): parse medibus data into waveform data
        medibus_data = reader.float32(  # noqa; variable will be used in future version
            length=52
        )

        self._parse_medibus_data(index, medibus_data, waveform_data)

        if index >= 0:
            # The event marker stays the same until the next event occurs. Therefore, check whether the
            # event marker has changed with respect to the most recent event. If so, create a new event.
            if (previous_marker is not None) and (event_marker > previous_marker):
                self.events.append(Event(index, current_time, event_marker, event_text))
            if timing_error:
                self.timing_errors.append(
                    TimingError(index, current_time, timing_error)
                )
            if min_max_flag == 1:
                self.phases.append(MaxValue(index, current_time))
            elif min_max_flag == -1:
                self.phases.append(MinValue(index, current_time))
            self.time[index] = current_time

        return event_marker

    def _parse_medibus_data(
        self, index: int, data: list, waveform_data: dict[str, NDArray]
    ):
        for key, value in zip(medibus_field_names, data):
            # some fields have a value of -1000 when not available; others a very large negative
            # number which varies, but seems to always be lower than -3e37
            if value == -1000:
                continue
            if value < -3e37:
                continue

            waveform_data[key][index] = value

    @staticmethod
    def reshape_frame(frame):
        """Convert linear array into 2D (32x32) image-like array."""
        return np.reshape(frame, (32, 32), "C")


@dataclass(eq=False)
class TimpelSequence(Sequence):
    """Sequence object for TIMPEL data."""

    vendor: Vendor = Vendor.TIMPEL

    def _load_data(self, first_frame: int):
        """Load data for TIMPEL files."""

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
                f"File {self.path} could not be read as Timpel data.\n"
                "Make sure this is a valid and uncorrupted Timpel data file.\n"
                f"Original error message: {e}"
            ) from e

        data: NDArray
        if data.shape[1] != COLUMN_WIDTH:
            raise OSError(
                f"Input does not have a width of {COLUMN_WIDTH} columns.\n"
                "Make sure this is a valid and uncorrupted Timpel data file."
            )
        if data.shape[0] == 0:
            raise ValueError(
                f"Invalid input: `first_frame` {first_frame} is larger than the "
                f"total number of frames in the file."
            )
        if data.shape[0] != self.nframes:
            if self.nframes:
                warnings.warn(
                    f"The number of frames requested ({self.nframes}) is larger "
                    f"than the available number ({data.shape[0]}) of frames after "
                    f"the first frame selected ({first_frame}).\n"
                    f"{data.shape[0]} frames have been loaded."
                )
            self.nframes = data.shape[0]

        # TODO (#80): QUESTION: check whether below issue was only a Drager problem or also
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


medibus_field_names = [
    "airway pressure [mbar]",
    "flow [L/min]",
    "volume [mL]",
    "CO2 [%]",
    "CO2 [kPa]",
    "CO2 [mmHg]",
    "dynamic compliance [mL/mbar]",
    "resistance [mbar/L/s]",
    "r² ???",
    "spontaneous inspiratory time [s]",
    "minimal pressure [mbar]",
    "P0.1 [mbar]",
    "mean pressure [mbar]",
    "plateau pressure [mbar]",
    "PEEP [mbar]",
    "intrinsic PEEP [mbar]",
    "mandatory respiratory rate [/min]",
    "mandatory minute volume [L/min]",
    "peak inspiratory pressure [mbar]",
    "mandatory tidal volume [L]",
    "spontaneous tidal volume [L]",
    "trapped volume [mL]",
    "mandatory expiratory tidal volume [mL]",
    "spontaneous expiratory tidal volume [mL]",
    "mandatory inspiratory tidal volume [mL]",
    "tidal volume [mL]",
    "spontaneous inspiratory tidal volume [mL]",
    "negative inspiratory force [mbar]",
    "leak minute volume [L/min]",
    "leak percentage [%]",
    "spontaneous respiratory rate [/min]",
    "percentage of spontaneous minute volume [%]",
    "spontaneous minute volume [L/min]",
    "minute volume [L/min]",
    "airway temperature [°C]",
    "rapid shallow breating index [1/min/L]",
    "respiratory rate [/min]",
    "inspiratory:expiratory ratio",
    "CO2 flow [mL/min]",
    "dead space volume [mL]",
    "percentage dead space of expiratory tidal volume [%]",
    "end-tidal CO2 [%]",
    "end-tidal CO2 [kPa]",
    "end-tidal CO2 [mmHg]",
    "fraction inspired O2 [%]",
    "spontaneous inspiratory:expiratory ratio",
    "elastance [mbar/L]",
    "TC ??? [s]",
    "ratio between upper 20% pressure range and total dynamic compliance",
    "end-inspiratory pressure [mbar]",
    "expiratory tidal volume [mL]",
    "time at low pressure [s]",
]
