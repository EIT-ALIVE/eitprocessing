from __future__ import annotations

import warnings
from collections import deque
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from eitprocessing.datahandling.breath import Breath
from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sparsedata import SparseData

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_COLUMN_WIDTH = 1030
_NAN_VALUE = -1000

TIMPEL_FRAMERATE = 50


load_timpel_data = partial(load_eit_data, vendor=Vendor.TIMPEL)


def load_from_single_path(
    path: Path,
    framerate: float | None = 20,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> DataCollection | tuple[DataCollection, DataCollection, DataCollection]:
    """Load Timpel EIT data from path."""
    if not framerate:
        framerate = TIMPEL_FRAMERATE

    try:
        data: NDArray = np.loadtxt(
            str(path),
            dtype=float,
            delimiter=",",
            skiprows=first_frame,
            max_rows=max_frames,
        )
    except UnicodeDecodeError as e:
        msg = (
            f"File {path} could not be read as Timpel data.\n"
            "Make sure this is a valid and uncorrupted Timpel data file.\n"
            f"Original error message: {e}"
        )
        raise OSError(msg) from e

    if data.shape[1] != _COLUMN_WIDTH:
        msg = (
            f"Input does not have a width of {_COLUMN_WIDTH} columns.\n"
            "Make sure this is a valid and uncorrupted Timpel data file."
        )
        raise OSError(msg)
    if data.shape[0] == 0:
        msg = f"Invalid input: `first_frame` {first_frame} is larger than the total number of frames in the file."
        raise ValueError(msg)

    if max_frames and data.shape[0] == max_frames:
        nframes = max_frames
    else:
        if max_frames:
            warnings.warn(
                f"The number of frames requested ({max_frames}) is larger "
                f"than the available number ({data.shape[0]}) of frames after "
                f"the first frame selected ({first_frame}).\n"
                f"{data.shape[0]} frames have been loaded.",
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

    pixel_impedance = np.where(pixel_impedance == _NAN_VALUE, np.nan, pixel_impedance)

    eit_data = EITData(
        vendor=Vendor.TIMPEL,
        label="raw",
        path=path,
        nframes=nframes,
        time=time,
        framerate=framerate,
        pixel_impedance=pixel_impedance,
    )
    eitdata_collection = DataCollection(EITData)
    eitdata_collection.add(eit_data)

    # extract waveform data
    # TODO: properly export waveform data

    continuousdata_collection = DataCollection(ContinuousData)
    continuousdata_collection.add(
        ContinuousData(
            "global_impedance_(raw)",
            "Global impedance",
            "a.u.",
            "global_impedance",
            "Global impedance calculated from raw EIT data",
            time=time,
            values=eit_data._calculate_global_impedance(),  # noqa: SLF001
        ),
    )
    continuousdata_collection.add(
        ContinuousData(
            "airway_pressure_(timpel)",
            "Airway pressure",
            "cmH2O",
            "pressure",
            "Airway pressure measured by Timpel device",
            time=time,
            values=data[:, 1024],
        ),
    )

    continuousdata_collection.add(
        ContinuousData(
            "flow_(timpel)",
            "Flow",
            "L/s",
            "flow",
            "FLow measures by Timpel device",
            time=time,
            values=data[:, 1025],
        ),
    )

    continuousdata_collection.add(
        ContinuousData(
            "volume_(timpel)",
            "Volume",
            "L",
            "volume",
            "Volume measured by Timpel device",
            time=time,
            values=data[:, 1026],
        ),
    )

    eit_data = EITData(
        vendor=Vendor.TIMPEL,
        label="raw",
        path=path,
        nframes=nframes,
        time=time,
        framerate=framerate,
        pixel_impedance=pixel_impedance,
    )
    eit_data_collection = DataCollection(EITData)
    eit_data_collection.add(eit_data)

    sparsedata_collection = DataCollection(SparseData)

    # extract breath start, breath end and QRS marks
    min_indices = np.nonzero(data[:, 1027] == 1)[0]
    sparsedata_collection.add(
        SparseData(
            label="minvalues_(timpel)",
            name="Minimum values detected by Timpel device.",
            unit=None,
            category="minvalue",
            derived_from=[eit_data],
            time=time[min_indices],
        ),
    )

    max_indices = np.nonzero(data[:, 1028] == 1)[0]
    sparsedata_collection.add(
        SparseData(
            label="maxvalues_(timpel)",
            name="Maximum values detected by Timpel device.",
            unit=None,
            category="maxvalue",
            derived_from=[eit_data],
            time=time[max_indices],
        ),
    )

    qrs_indices = np.nonzero(data[:, 1029] == 1)[0]
    sparsedata_collection.add(
        SparseData(
            label="qrscomplexes_(timpel)",
            name="QRS complexes detected by Timpel device",
            unit=None,
            category="qrs_complex",
            derived_from=[eit_data],
            time=time[qrs_indices],
        ),
    )

    return eit_data_collection, continuous_data_collection, sparsedata_collection
