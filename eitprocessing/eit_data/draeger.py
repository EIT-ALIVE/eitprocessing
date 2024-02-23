import sys
import warnings
from collections import namedtuple
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self
from eitprocessing.continuous_data.continuous_data_collection import (
    ContinuousDataCollection,
)
from eitprocessing.continuous_data.continuous_data_variant import ContinuousDataVariant
from eitprocessing.sparse_data.sparse_data_collection import SparseDataCollection
from ..binreader.reader import Reader
from ..continuous_data import ContinuousData
from . import EITData_
from .eit_data_variant import EITDataVariant
from .event import Event
from .phases import MaxValue
from .phases import MinValue
from .vendor import Vendor


@dataclass(eq=False)
class DraegerEITData(EITData_):
    """Container for EIT data recorded using the Dräger Pulmovista PV500."""

    vendor: Vendor = field(default=Vendor.DRAEGER, init=False)
    framerate: float = 20

    @classmethod
    def _from_path(  # pylint: disable=too-many-arguments,too-many-locals
        cls,
        path: Path,
        label: str | None = None,
        framerate: float | None = 20,
        first_frame: int = 0,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> Self | tuple[Self, ContinuousDataCollection, SparseDataCollection]:
        """"""

        FRAME_SIZE_BYTES = 4358

        file_size = path.stat().st_size
        if file_size % FRAME_SIZE_BYTES:
            raise OSError(
                f"File size {file_size} of file {str(path)} not divisible by "
                f"{FRAME_SIZE_BYTES}.\n"
                f"Make sure this is a valid and uncorrupted Dräger data file."
            )
        total_frames = file_size // FRAME_SIZE_BYTES

        if first_frame > total_frames:
            raise ValueError(
                f"Invalid input: `first_frame` {first_frame} is larger than the "
                f"total number of frames in the file {total_frames}."
            )

        n_frames = min(total_frames - first_frame, max_frames or sys.maxsize)

        if max_frames and max_frames != n_frames:
            warnings.warn(
                f"The number of frames requested ({max_frames}) is larger "
                f"than the available number ({n_frames}) of frames after "
                f"the first frame selected ({first_frame}, total frames: "
                f"{total_frames}).\n {n_frames} frames will be loaded."
            )

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
            fh.seek(first_frame_to_load * FRAME_SIZE_BYTES)
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

        obj = cls(
            path=path,
            framerate=framerate,
            nframes=n_frames,
            time=time,
            phases=phases,
            events=events,
            label=label,
        )
        obj.variants.add(
            EITDataVariant(
                label="raw",
                description="raw impedance data",
                pixel_impedance=pixel_impedance,
            )
        )
        if return_non_eit_data:
            (
                continuous_data_coll,
                sparse_data_coll,
            ) = cls._convert_medibus_data(medibus_data, time)

            return (obj, continuous_data_coll, sparse_data_coll)

        return obj

    @classmethod
    def _convert_medibus_data(
        cls, medibus_data: NDArray, time: NDArray
    ) -> tuple[ContinuousDataCollection, SparseDataCollection]:
        continuous_data_collection = ContinuousDataCollection()
        sparse_data_collection = SparseDataCollection()

        for field_info, data in zip(medibus_fields, medibus_data):
            if field_info.continuous:
                continuous_data = ContinuousData(
                    name=field_info.signal_name,
                    description=f"continuous {field_info.signal_name} data loaded from file",
                    unit=field_info.unit,
                    time=time,
                    loaded=True,
                    category=field_info.unit,
                )
                continuous_data.variants.add(
                    ContinuousDataVariant(
                        label="raw",
                        description="raw data loaded from file",
                        values=data,
                    )
                )
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


medibus_field = namedtuple("medibus_field", ["signal_name", "unit", "continuous"])

medibus_fields = [
    medibus_field("airway pressure", "mbar", True),
    medibus_field("flow", "L/min", True),
    medibus_field("volume", "mL", True),
    medibus_field("CO2", "%", True),
    medibus_field("CO2", "kPa", True),
    medibus_field("CO2", "mmHg", True),
    medibus_field("dynamic compliance", "mL/mbar", False),
    medibus_field("resistance", "mbar/L/s", False),
    medibus_field("r^2", "", False),
    medibus_field("spontaneous inspiratory time", "s", False),
    medibus_field("minimal pressure", "mbar", False),
    medibus_field("P0.1", "mbar", False),
    medibus_field("mean pressure", "mbar", False),
    medibus_field("plateau pressure", "mbar", False),
    medibus_field("PEEP", "mbar", False),
    medibus_field("intrinsic PEEP", "mbar", False),
    medibus_field("mandatory respiratory rate", "/min", False),
    medibus_field("mandatory minute volume", "L/min", False),
    medibus_field("peak inspiratory pressure", "mbar", False),
    medibus_field("mandatory tidal volume", "L", False),
    medibus_field("spontaneous tidal volume", "L", False),
    medibus_field("trapped volume", "mL", False),
    medibus_field("mandatory expiratory tidal volume", "mL", False),
    medibus_field("spontaneous expiratory tidal volume", "mL", False),
    medibus_field("mandatory inspiratory tidal volume", "mL", False),
    medibus_field("tidal volume", "mL", False),
    medibus_field("spontaneous inspiratory tidal volume", "mL", False),
    medibus_field("negative inspiratory force", "mbar", False),
    medibus_field("leak minute volume", "L/min", False),
    medibus_field("leak percentage", "%", False),
    medibus_field("spontaneous respiratory rate", "/min", False),
    medibus_field("percentage of spontaneous minute volume", "%", False),
    medibus_field("spontaneous minute volume", "L/min", False),
    medibus_field("minute volume", "L/min", False),
    medibus_field("airway temperature", "degrees C", False),
    medibus_field("rapid shallow breating index", "1/min/L", False),
    medibus_field("respiratory rate", "/min", False),
    medibus_field("inspiratory:expiratory ratio", "", False),
    medibus_field("CO2 flow", "mL/min", False),
    medibus_field("dead space volume", "mL", False),
    medibus_field("percentage dead space of expiratory tidal volume", "%", False),
    medibus_field("end-tidal CO2", "%", False),
    medibus_field("end-tidal CO2", "kPa", False),
    medibus_field("end-tidal CO2", "mmHg", False),
    medibus_field("fraction inspired O2", "%", False),
    medibus_field("spontaneous inspiratory:expiratory ratio", "", False),
    medibus_field("elastance", "mbar/L", False),
    medibus_field("time constant", "s", False),
    medibus_field(
        "ratio between upper 20% pressure range and total dynamic compliance", "", False
    ),
    medibus_field("end-inspiratory pressure", "mbar", False),
    medibus_field("expiratory tidal volume", "mL", False),
    medibus_field("time at low pressure", "s", False),
]
