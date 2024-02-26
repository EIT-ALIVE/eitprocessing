from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from eitprocessing.binreader.reader import Reader
from eitprocessing.continuous_data import ContinuousData
from eitprocessing.data_collection import DataCollection
from eitprocessing.sparse_data import SparseData

from . import EITData_
from .event import Event
from .phases import MaxValue, MinValue
from .vendor import Vendor

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_FRAME_SIZE_BYTES = 4358


@dataclass(eq=False)
class DraegerEITData(EITData_):
    """Container for EIT data recorded using the Dräger Pulmovista PV500."""

    vendor: Vendor = field(default=Vendor.DRAEGER, init=False)
    framerate: float = field(default=20, metadata={"check_equivalence_equals": True, "concatenate": "first"})

    @classmethod
    def _from_path(  # pylint: disable=too-many-arguments,too-many-locals
        cls,
        path: Path,
        framerate: float | None = 20,
        first_frame: int = 0,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> DataCollection | tuple[DataCollection, DataCollection, DataCollection]:
        file_size = path.stat().st_size
        if file_size % _FRAME_SIZE_BYTES:
            msg = (
                f"File size {file_size} of file {path!s} not divisible by "
                f"{_FRAME_SIZE_BYTES}.\n"
                f"Make sure this is a valid and uncorrupted Dräger data file."
            )
            raise OSError(msg)
        total_frames = file_size // _FRAME_SIZE_BYTES

        if first_frame > total_frames:
            msg = (
                f"Invalid input: `first_frame` {first_frame} is larger than the "
                f"total number of frames in the file {total_frames}."
            )
            raise ValueError(msg)

        n_frames = min(total_frames - first_frame, max_frames or sys.maxsize)

        if max_frames and max_frames != n_frames:
            msg = (
                f"The number of frames requested ({max_frames}) is larger "
                f"than the available number ({n_frames}) of frames after "
                f"the first frame selected ({first_frame}, total frames: "
                f"{total_frames}).\n {n_frames} frames will be loaded.",
            )
            warnings.warn(msg)

        # We need to load 1 frame before first actual frame to check if there
        # is an event marker. Data for the pre-first (dummy) frame will be
        # removed from self at the end of this function.
        load_dummy_frame = first_frame > 0
        first_frame_to_load = first_frame - 1 if load_dummy_frame else 0

        pixel_impedance = np.zeros((n_frames, 32, 32))
        time = np.zeros((n_frames,))
        events = []
        phases = []
        medibus_data = np.zeros((52, n_frames))

        with open(path, "br") as fh:
            fh.seek(first_frame_to_load * _FRAME_SIZE_BYTES)
            reader = Reader(fh)
            previous_marker = None

            first_index = -1 if load_dummy_frame else 0
            for index in range(first_index, n_frames):
                previous_marker = cls._read_frame(
                    reader,
                    index,
                    time,
                    pixel_impedance,
                    medibus_data,
                    events,
                    phases,
                    previous_marker,
                )

        if not framerate:
            framerate = cls.framerate

        eit_data_collection = DataCollection(cls)
        eit_data_collection.add(
            cls(
                path=path,
                framerate=framerate,
                nframes=n_frames,
                time=time,
                phases=phases,
                events=events,
                label="raw",
                pixel_impedance=pixel_impedance,
            ),
        )
        if return_non_eit_data:
            (
                continuous_data_collection,
                sparse_data_collections,
            ) = cls._convert_medibus_data(medibus_data)

            return (eit_data_collection, continuous_data_collection, sparse_data_collections)

        return eit_data_collection

    @classmethod
    def _convert_medibus_data(
        cls,
        medibus_data: NDArray,
    ) -> tuple[DataCollection, DataCollection]:
        continuous_data_collection = DataCollection(ContinuousData)
        sparse_data_collection = DataCollection(SparseData)

        for field_info, data in zip(medibus_fields, medibus_data):
            if field_info.continuous:
                continuous_data = ContinuousData(
                    label=field_info.signal_name,
                    name=field_info.signal_name,
                    description=f"Continuous {field_info.signal_name} data loaded from file",
                    unit=field_info.unit,
                    loaded=True,
                    values=data,
                    category=field_info.signal_name,
                )
                continuous_data.lock()
                continuous_data_collection.add(continuous_data)

            else:
                # TODO parse sparse data
                ...

        return continuous_data_collection, sparse_data_collection

    @classmethod
    def _read_frame(  # pylint: disable=too-many-arguments,too-many-locals
        cls,
        reader: Reader,
        index: int,
        time: NDArray,
        pixel_impedance: NDArray,
        medibus_data: NDArray,
        events: list,
        phases: list,
        previous_marker: int | None,
    ) -> int:
        """Read frame by frame data from DRAEGER files.

        This method adds the loaded data to the provided arrays `time` and
        `pixel_impedance` and the provided lists `events` and `phases` when the
        index is non-negative. When the index is negative, no data is saved. In
        any case, the event marker is returned.
        """
        frame_time = round(reader.float64() * 24 * 60 * 60, 3)
        _ = reader.float32()
        frame_pixel_impedance = reader.npfloat32(length=1024)
        frame_pixel_impedance = np.reshape(frame_pixel_impedance, (32, 32), "C")
        min_max_flag = reader.int32()
        event_marker = reader.int32()
        event_text = reader.string(length=30)
        timing_error = reader.int32()

        frame_medibus_data = reader.npfloat32(length=52)  # noqa;

        if index < 0:
            # do not keep any loaded data, just return the event marker
            return event_marker

        time[index] = frame_time
        pixel_impedance[index, :, :] = frame_pixel_impedance
        medibus_data[:, index] = frame_medibus_data

        # The event marker stays the same until the next event occurs.
        # Therefore, check whether the event marker has changed with
        # respect to the most recent event. If so, create a new event.
        if (previous_marker is not None) and (event_marker > previous_marker):
            events.append(Event(index, frame_time, event_marker, event_text))
        if timing_error:
            warnings.warn("A timing error was encountered during loading.")
            # TODO: expand on what timing errors are in some documentation.
        if min_max_flag == 1:
            phases.append(MaxValue(index, frame_time))
        elif min_max_flag == -1:
            phases.append(MinValue(index, frame_time))

        return event_marker


class MedibusField(NamedTuple):
    signal_name: str
    unit: str
    continuous: bool


medibus_fields = [
    MedibusField("airway pressure", "mbar", True),
    MedibusField("flow", "L/min", True),
    MedibusField("volume", "mL", True),
    MedibusField("CO2", "%", True),
    MedibusField("CO2", "kPa", True),
    MedibusField("CO2", "mmHg", True),
    MedibusField("dynamic compliance", "mL/mbar", False),
    MedibusField("resistance", "mbar/L/s", False),
    MedibusField("r^2", "", False),
    MedibusField("spontaneous inspiratory time", "s", False),
    MedibusField("minimal pressure", "mbar", False),
    MedibusField("P0.1", "mbar", False),
    MedibusField("mean pressure", "mbar", False),
    MedibusField("plateau pressure", "mbar", False),
    MedibusField("PEEP", "mbar", False),
    MedibusField("intrinsic PEEP", "mbar", False),
    MedibusField("mandatory respiratory rate", "/min", False),
    MedibusField("mandatory minute volume", "L/min", False),
    MedibusField("peak inspiratory pressure", "mbar", False),
    MedibusField("mandatory tidal volume", "L", False),
    MedibusField("spontaneous tidal volume", "L", False),
    MedibusField("trapped volume", "mL", False),
    MedibusField("mandatory expiratory tidal volume", "mL", False),
    MedibusField("spontaneous expiratory tidal volume", "mL", False),
    MedibusField("mandatory inspiratory tidal volume", "mL", False),
    MedibusField("tidal volume", "mL", False),
    MedibusField("spontaneous inspiratory tidal volume", "mL", False),
    MedibusField("negative inspiratory force", "mbar", False),
    MedibusField("leak minute volume", "L/min", False),
    MedibusField("leak percentage", "%", False),
    MedibusField("spontaneous respiratory rate", "/min", False),
    MedibusField("percentage of spontaneous minute volume", "%", False),
    MedibusField("spontaneous minute volume", "L/min", False),
    MedibusField("minute volume", "L/min", False),
    MedibusField("airway temperature", "degrees C", False),
    MedibusField("rapid shallow breating index", "1/min/L", False),
    MedibusField("respiratory rate", "/min", False),
    MedibusField("inspiratory:expiratory ratio", "", False),
    MedibusField("CO2 flow", "mL/min", False),
    MedibusField("dead space volume", "mL", False),
    MedibusField("percentage dead space of expiratory tidal volume", "%", False),
    MedibusField("end-tidal CO2", "%", False),
    MedibusField("end-tidal CO2", "kPa", False),
    MedibusField("end-tidal CO2", "mmHg", False),
    MedibusField("fraction inspired O2", "%", False),
    MedibusField("spontaneous inspiratory:expiratory ratio", "", False),
    MedibusField("elastance", "mbar/L", False),
    MedibusField("time constant", "s", False),
    MedibusField(
        "ratio between upper 20% pressure range and total dynamic compliance",
        "",
        False,
    ),
    MedibusField("end-inspiratory pressure", "mbar", False),
    MedibusField("expiratory tidal volume", "mL", False),
    MedibusField("time at low pressure", "s", False),
]
