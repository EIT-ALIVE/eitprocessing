from __future__ import annotations

import sys
import warnings
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from eitprocessing.binreader.reader import Reader
from eitprocessing.continuous_data import ContinuousData
from eitprocessing.data_collection import DataCollection
from eitprocessing.eit_data import EITData
from eitprocessing.eit_data.event import Event
from eitprocessing.eit_data.loading import load_data
from eitprocessing.eit_data.phases import MaxValue, MinValue
from eitprocessing.eit_data.vendor import Vendor
from eitprocessing.sparse_data import SparseData

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_FRAME_SIZE_BYTES = 4358
DRAEGER_FRAMERATE = 20
load_draeger_data = partial(load_data, vendor=Vendor.DRAEGER)


def load_from_single_path(
    path: Path,
    framerate: float | None = 20,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> tuple[DataCollection, DataCollection, DataCollection]:
    """Load Dräger EIT data from path(s).

    See loading.from_path().
    """
    file_size = path.stat().st_size
    if file_size % _FRAME_SIZE_BYTES:
        msg = (
            f"File size {file_size} of file {path!s} not divisible by {_FRAME_SIZE_BYTES}.\n"
            f"Make sure this is a valid and uncorrupted Dräger data file."
        )
        raise OSError(msg)
    total_frames = file_size // _FRAME_SIZE_BYTES

    if (f0 := first_frame) > (fn := total_frames):
        msg = f"Invalid input: `first_frame` ({f0}) is larger than the total number of frames in the file ({fn})."
        raise ValueError(msg)

    n_frames = min(total_frames - first_frame, max_frames or sys.maxsize)

    if max_frames and max_frames != n_frames:
        msg = (
            f"The number of frames requested ({max_frames}) is larger "
            f"than the available number ({n_frames}) of frames after "
            f"the first frame selected ({first_frame}, total frames: "
            f"{total_frames}).\n {n_frames} frames will be loaded."
        )
        warnings.warn(msg)

    # We need to load 1 frame before first actual frame to check if there is an event marker. Data for the pre-first
    # (dummy) frame will be removed from self at the end of this function.
    load_dummy_frame = first_frame > 0
    first_frame_to_load = first_frame - 1 if load_dummy_frame else 0

    pixel_impedance = np.zeros((n_frames, 32, 32))
    time = np.zeros((n_frames,))
    events = []
    phases = []
    medibus_data = np.zeros((52, n_frames))

    with path.open("br") as fh:
        fh.seek(first_frame_to_load * _FRAME_SIZE_BYTES)
        reader = Reader(fh)
        previous_marker = None

        first_index = -1 if load_dummy_frame else 0
        for index in range(first_index, n_frames):
            previous_marker = _read_frame(
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
        framerate = DRAEGER_FRAMERATE

    eit_data_collection = DataCollection(EITData)
    eit_data_collection.add(
        EITData(
            vendor=Vendor.DRAEGER,
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
    (
        continuous_data_collection,
        sparse_data_collections,
    ) = _convert_medibus_data(medibus_data, time)

    return (
        eit_data_collection,
        continuous_data_collection,
        sparse_data_collections,
    )


def _convert_medibus_data(
    medibus_data: NDArray,
    time: NDArray,
) -> tuple[DataCollection, DataCollection]:
    continuous_data_collection = DataCollection(ContinuousData)
    sparse_data_collection = DataCollection(SparseData)

    for field_info, data in zip(_medibus_fields, medibus_data, strict=True):
        if field_info.continuous:
            continuous_data = ContinuousData(
                label=field_info.signal_name,
                name=field_info.signal_name,
                description=f"Continuous {field_info.signal_name} data loaded from file",
                unit=field_info.unit,
                time=time,
                values=data,
                category=field_info.signal_name,
            )
            continuous_data.lock()
            continuous_data_collection.add(continuous_data)

        else:
            # TODO parse sparse data
            ...

    return continuous_data_collection, sparse_data_collection


def _read_frame(  # noqa: PLR0913
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

    frame_medibus_data = reader.npfloat32(length=52)

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


class _MedibusField(NamedTuple):
    signal_name: str
    unit: str
    continuous: bool


_medibus_fields = [
    _MedibusField("airway pressure", "mbar", True),
    _MedibusField("flow", "L/min", True),
    _MedibusField("volume", "mL", True),
    _MedibusField("CO2", "%", True),
    _MedibusField("CO2", "kPa", True),
    _MedibusField("CO2", "mmHg", True),
    _MedibusField("dynamic compliance", "mL/mbar", False),
    _MedibusField("resistance", "mbar/L/s", False),
    _MedibusField("r^2", "", False),
    _MedibusField("spontaneous inspiratory time", "s", False),
    _MedibusField("minimal pressure", "mbar", False),
    _MedibusField("P0.1", "mbar", False),
    _MedibusField("mean pressure", "mbar", False),
    _MedibusField("plateau pressure", "mbar", False),
    _MedibusField("PEEP", "mbar", False),
    _MedibusField("intrinsic PEEP", "mbar", False),
    _MedibusField("mandatory respiratory rate", "/min", False),
    _MedibusField("mandatory minute volume", "L/min", False),
    _MedibusField("peak inspiratory pressure", "mbar", False),
    _MedibusField("mandatory tidal volume", "L", False),
    _MedibusField("spontaneous tidal volume", "L", False),
    _MedibusField("trapped volume", "mL", False),
    _MedibusField("mandatory expiratory tidal volume", "mL", False),
    _MedibusField("spontaneous expiratory tidal volume", "mL", False),
    _MedibusField("mandatory inspiratory tidal volume", "mL", False),
    _MedibusField("tidal volume", "mL", False),
    _MedibusField("spontaneous inspiratory tidal volume", "mL", False),
    _MedibusField("negative inspiratory force", "mbar", False),
    _MedibusField("leak minute volume", "L/min", False),
    _MedibusField("leak percentage", "%", False),
    _MedibusField("spontaneous respiratory rate", "/min", False),
    _MedibusField("percentage of spontaneous minute volume", "%", False),
    _MedibusField("spontaneous minute volume", "L/min", False),
    _MedibusField("minute volume", "L/min", False),
    _MedibusField("airway temperature", "degrees C", False),
    _MedibusField("rapid shallow breating index", "1/min/L", False),
    _MedibusField("respiratory rate", "/min", False),
    _MedibusField("inspiratory:expiratory ratio", "", False),
    _MedibusField("CO2 flow", "mL/min", False),
    _MedibusField("dead space volume", "mL", False),
    _MedibusField("percentage dead space of expiratory tidal volume", "%", False),
    _MedibusField("end-tidal CO2", "%", False),
    _MedibusField("end-tidal CO2", "kPa", False),
    _MedibusField("end-tidal CO2", "mmHg", False),
    _MedibusField("fraction inspired O2", "%", False),
    _MedibusField("spontaneous inspiratory:expiratory ratio", "", False),
    _MedibusField("elastance", "mbar/L", False),
    _MedibusField("time constant", "s", False),
    _MedibusField(
        "ratio between upper 20% pressure range and total dynamic compliance",
        "",
        False,
    ),
    _MedibusField("end-inspiratory pressure", "mbar", False),
    _MedibusField("expiratory tidal volume", "mL", False),
    _MedibusField("time at low pressure", "s", False),
]
