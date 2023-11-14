import sys
import warnings
from dataclasses import dataclass
from dataclasses import field
from os import PathLike
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self
from typing_extensions import override
from ..binreader.reader import Reader
from ..variants.variant_collection import VariantCollection
from . import EITData
from .eit_data_variant import EITDataVariant
from .event import Event
from .phases import MaxValue
from .phases import MinValue
from .vendor import Vendor


@dataclass(eq=False)
class DraegerEITData(EITData):
    """Container for EIT data recorded using the Dräger Pulmovista PV500."""

    vendor: Vendor = field(default=Vendor.DRAEGER, init=False)
    framerate: float = 20
    variants: VariantCollection = field(
        default_factory=lambda: VariantCollection(EITDataVariant)
    )

    @override
    @classmethod
    def from_path(
        cls,
        path: PathLike | list[PathLike],
        label: str | None = None,
        framerate: int | None = None,
        first_frame: int = 0,
        max_frames: int | None = None,
    ) -> Self:
        return super().from_path(
            path, cls.vendor, label, framerate, first_frame, max_frames
        )

    @classmethod
    def _from_path(
        cls,
        path: Path,
        label: str | None,
        framerate: float | None = 20,
        first_frame: int = 0,
        max_frames: int | None = None,
    ) -> Self:
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
                    events,
                    phases,
                    previous_marker,
                )

        if not framerate:
            framerate = cls.framerate

        obj = cls(path=path, framerate=framerate, nframes=n_frames, time=time)
        obj.variants.add(
            EITDataVariant(
                name="raw",
                description="raw impedance data",
                pixel_impedance=pixel_impedance,
            )
        )

        return obj

    @classmethod
    def _read_frame(
        cls,
        reader: Reader,
        index: int,
        time: NDArray,
        pixel_impedance: NDArray,
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
        frame_pixel_impedance = cls._reshape_frame(reader.npfloat32(length=1024))
        min_max_flag = reader.int32()
        event_marker = reader.int32()
        event_text = reader.string(length=30)
        timing_error = reader.int32()

        # TODO (#79): parse medibus data into waveform data
        medibus_data = reader.npfloat32(
            length=52
        )  # noqa; variable will be used in future version

        if index < 0:
            # do not keep any loaded data, just return the event marker
            return event_marker

        time[index] = frame_time
        pixel_impedance[index, :, :] = frame_pixel_impedance

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

    @staticmethod
    def _reshape_frame(frame):
        """Convert linear array into 2D (32x32) image-like array."""
        return np.reshape(frame, (32, 32), "C")
