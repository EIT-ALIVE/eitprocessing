import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from eitprocessing.continuous_data.continuous_data_collection import (
    ContinuousDataCollection,
)
from eitprocessing.eit_data import EITData_
from eitprocessing.eit_data.eit_data_variant import EITDataVariant
from eitprocessing.eit_data.phases import MaxValue, MinValue, QRSMark
from eitprocessing.eit_data.vendor import Vendor
from eitprocessing.sparse_data.sparse_data_collection import SparseDataCollection
from eitprocessing.variants.variant_collection import VariantCollection


@dataclass(eq=False)
class TimpelEITData(EITData_):
    framerate: float = 50
    vendor: Vendor = field(default=Vendor.TIMPEL, init=False)
    variants: VariantCollection = field(
        default_factory=lambda: VariantCollection(EITDataVariant)
    )

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
        COLUMN_WIDTH = 1030

        if not framerate:
            framerate = cls.framerate

        try:
            data: NDArray = np.loadtxt(
                str(path),
                dtype=float,
                delimiter=",",
                skiprows=first_frame,
                max_rows=max_frames,
            )
        except UnicodeDecodeError as e:
            raise OSError(
                f"File {path} could not be read as Timpel data.\n"
                "Make sure this is a valid and uncorrupted Timpel data file.\n"
                f"Original error message: {e}"
            ) from e

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

        if max_frames and data.shape[0] == max_frames:
            nframes = max_frames
        else:
            if max_frames:
                warnings.warn(
                    f"The number of frames requested ({max_frames}) is larger "
                    f"than the available number ({data.shape[0]}) of frames after "
                    f"the first frame selected ({first_frame}).\n"
                    f"{data.shape[0]} frames have been loaded."
                )
            nframes = data.shape[0]

        # TODO (#80): QUESTION: check whether below issue was only a Drager problem or also
        # applicable to Timpel.
        # The implemented method seems convoluted: it's easier to create an array
        # with nframes and add a time_offset. However, this results in floating
        # point errors, creating issues with comparing times later on.
        time = np.arange(nframes + first_frame) / framerate
        time = time[first_frame:]

        pixel_impedance = data[:, :1024]
        pixel_impedance = np.reshape(pixel_impedance, newshape=(-1, 32, 32), order="C")
        pixel_impedance = np.where(pixel_impedance == -1000, np.nan, pixel_impedance)

        # extract waveform data
        # TODO: properly export waveform data
        waveform_data = {  # noqa
            "airway_pressure": data[:, 1024],
            "flow": data[:, 1025],
            "volume": data[:, 1026],
        }

        # extract breath start, breath end and QRS marks
        phases = []
        for index in np.flatnonzero(data[:, 1027] == 1):
            phases.append(MinValue(index, time[int(index)]))

        for index in np.flatnonzero(data[:, 1028] == 1):
            phases.append(MaxValue(index, time[int(index)]))

        for index in np.flatnonzero(data[:, 1029] == 1):
            phases.append(QRSMark(index, time[int(index)]))

        phases.sort(key=lambda x: x.index)

        obj = cls(
            path=path,
            nframes=nframes,
            time=time,
            framerate=framerate,
            phases=phases,
            label=label,
        )
        obj.variants.add(
            EITDataVariant(
                label="raw",
                description="raw impedance data",
                pixel_impedance=pixel_impedance,
            )
        )

        return obj
