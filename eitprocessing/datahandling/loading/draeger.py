from __future__ import annotations

import mmap
import sys
import warnings
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.event import Event
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.loading.binreader import BinReader
from eitprocessing.datahandling.sparsedata import SparseData

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

load_draeger_data = partial(load_eit_data, vendor=Vendor.DRAEGER)
NAN_VALUE_INDICATOR = -1e30


def load_from_single_path(
    path: Path,
    sample_frequency: float | None = None,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> dict[str, DataCollection]:
    """Load Dräger EIT data from path."""
    file_size = path.stat().st_size

    frame_size: int
    medibus_fields: list

    # iterate over the supported file formats to find the frame size that matches the file size
    for _file_format_data in _bin_file_formats.values():
        frame_size = _file_format_data["frame_size"]
        if file_size % frame_size == 0:
            # if the file size is an integer multiple of the frame size, assume this is the correct format
            medibus_fields = _file_format_data["medibus_fields"]
            break
    else:
        msg = (
            f"File size {file_size} of file {path!s} does not match the supported *.bin file formats.\n"
            "Currently this package does not support loading files containing "
            "esophageal pressure or other non-standard data. "
            "Make sure this is a valid and uncorrupted Dräger data file."
        )
        raise OSError(msg)
    total_frames = file_size // frame_size

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
    events: list[tuple[float, Event]] = []
    phases: list[tuple[float, int]] = []
    medibus_data = np.zeros((len(medibus_fields), n_frames), dtype=np.float32)

    with path.open("br") as fo, mmap.mmap(fo.fileno(), length=0, access=mmap.ACCESS_READ) as fh:
        fh.seek(first_frame_to_load * frame_size)
        reader = BinReader(fh)
        previous_marker = None

        first_index = -1 if load_dummy_frame else 0
        for index in range(first_index, n_frames):
            previous_marker = _read_frame(
                reader,
                index,
                time,
                pixel_impedance,
                medibus_data,
                len(medibus_fields),
                events,
                phases,
                previous_marker,
            )

    # time wraps around the number of seconds in a day
    time = np.unwrap(time, period=24 * 60 * 60)

    sample_frequency = _estimate_sample_frequency(time, sample_frequency)

    eit_data = EITData(
        vendor=Vendor.DRAEGER,
        path=path,
        sample_frequency=sample_frequency,
        nframes=n_frames,
        time=time,
        label="raw",
        pixel_impedance=pixel_impedance,
    )
    eitdata_collection = DataCollection(EITData, raw=eit_data)

    (
        continuousdata_collection,
        sparsedata_collection,
    ) = _convert_medibus_data(medibus_data, medibus_fields, time, sample_frequency)
    intervaldata_collection = DataCollection(IntervalData)
    # TODO: move some medibus data to sparse / interval
    # TODO: move phases and events to sparse / interval

    continuousdata_collection.add(
        ContinuousData(
            label="global_impedance_(raw)",
            name="Global impedance (raw)",
            unit="a.u.",
            category="impedance",
            derived_from=[eit_data],
            time=eit_data.time,
            values=eit_data.calculate_global_impedance(),
            sample_frequency=sample_frequency,
        ),
    )
    sparsedata_collection.add(
        SparseData(
            label="minvalues_(draeger)",
            name="Minimum values detected by Draeger device.",
            unit=None,
            category="minvalue",
            derived_from=[eit_data],
            time=np.array([t for t, d in phases if d == -1]),
        ),
    )
    sparsedata_collection.add(
        SparseData(
            label="maxvalues_(draeger)",
            name="Maximum values detected by Draeger device.",
            unit=None,
            category="maxvalue",
            derived_from=[eit_data],
            time=np.array([t for t, d in phases if d == 1]),
        ),
    )
    if events:
        time_, events_ = zip(*events, strict=True)
        time = np.array(time_)
        events = list(events_)
    else:
        time, events = np.array([]), []
    sparsedata_collection.add(
        SparseData(
            label="events_(draeger)",
            name="Events loaded from Draeger data",
            unit=None,
            category="event",
            derived_from=[eit_data],
            time=time,
            values=events,
        ),
    )

    return {
        "eitdata_collection": eitdata_collection,
        "continuousdata_collection": continuousdata_collection,
        "sparsedata_collection": sparsedata_collection,
        "intervaldata_collection": intervaldata_collection,
    }


def _estimate_sample_frequency(time: np.ndarray, sample_frequency: float | None) -> float:
    """Estimate the sample frequency from the time axis, and check with provided sample frequency."""
    estimated_sample_frequency = round((len(time) - 1) / (time[-1] - time[0]), 4)

    if sample_frequency is None:
        return estimated_sample_frequency

    if sample_frequency != estimated_sample_frequency:
        msg = (
            f"Provided sample frequency ({sample_frequency}) does not match "
            f"the estimated sample frequency ({estimated_sample_frequency})."
        )
        warnings.warn(msg, RuntimeWarning)

    return sample_frequency


def _convert_medibus_data(
    medibus_data: NDArray,
    medibus_fields: list,
    time: NDArray,
    sample_frequency: float,
) -> tuple[DataCollection, DataCollection]:
    continuousdata_collection = DataCollection(ContinuousData)
    sparsedata_collection = DataCollection(SparseData)

    for field_info, data in zip(medibus_fields, medibus_data, strict=True):
        data[data < NAN_VALUE_INDICATOR] = np.nan
        if field_info.continuous:
            continuous_data = ContinuousData(
                label=field_info.signal_name,
                name=field_info.signal_name,
                description=f"Continuous {field_info.signal_name} data loaded from file",
                unit=field_info.unit,
                time=time,
                values=data,
                category=field_info.signal_name,
                sample_frequency=sample_frequency,
            )
            continuous_data.lock()
            continuousdata_collection.add(continuous_data)

        else:
            # TODO parse sparse data
            ...

    return continuousdata_collection, sparsedata_collection


def _read_frame(
    reader: BinReader,
    index: int,
    time: NDArray,
    pixel_impedance: NDArray,
    medibus_data: NDArray,
    n_medibus_fields: int,
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

    frame_medibus_data = reader.npfloat32(length=n_medibus_fields)

    if index < 0:
        # do not keep any loaded data, just return the event marker
        return event_marker

    time[index] = frame_time
    pixel_impedance[index, :, :] = frame_pixel_impedance
    medibus_data[:, index] = frame_medibus_data

    # The event marker stays the same until the next event occurs.
    # Therefore, check whether the event marker has changed with
    # respect to the most recent event. If so, create a new event.
    if ((previous_marker is not None) and (event_marker > previous_marker)) or (index == 0 and event_text):
        events.append((frame_time, Event(event_marker, event_text)))
    if timing_error:
        warnings.warn("A timing error was encountered during loading.")
        # TODO: expand on what timing errors are in some documentation.
    if min_max_flag in (1, -1):
        phases.append((frame_time, min_max_flag))

    return event_marker


class _MedibusField(NamedTuple):
    signal_name: str
    unit: str
    continuous: bool


_bin_file_formats = {
    "original": {
        "frame_size": 4358,
        "medibus_fields": [
            _MedibusField("airway pressure", "mbar", True),
            _MedibusField("flow", "L/min", True),
            _MedibusField("volume", "mL", True),
            _MedibusField("CO2 (%)", "%", True),
            _MedibusField("CO2 (kPa)", "kPa", True),
            _MedibusField("CO2 (mmHg)", "mmHg", True),
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
        ],
    },
    "pressure_pod": {
        "frame_size": 4382,
        "medibus_fields": [
            _MedibusField("airway pressure", "mbar", True),
            _MedibusField("flow", "L/min", True),
            _MedibusField("volume", "mL", True),
            _MedibusField("CO2 (%)", "%", True),
            _MedibusField("CO2 (kPa)", "kPa", True),
            _MedibusField("CO2 (mmHg)", "mmHg", True),
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
            _MedibusField("high pressure", "mbar", False),
            _MedibusField("low pressure", "mbar", False),
            _MedibusField("time at low pressure", "s", False),
            _MedibusField("airway pressure (pod)", "mbar", True),
            _MedibusField("esophageal pressure (pod)", "mbar", True),
            _MedibusField("transpulmonary pressure (pod)", "mbar", True),
            _MedibusField("gastric pressure/auxiliary pressure (pod)", "mbar", True),
        ],
    },
}
