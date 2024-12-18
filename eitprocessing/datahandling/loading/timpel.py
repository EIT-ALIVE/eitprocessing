from __future__ import annotations

import warnings
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

TIMPEL_SAMPLE_FREQUENCY = 50


load_timpel_data = partial(load_eit_data, vendor=Vendor.TIMPEL)


def load_from_single_path(
    path: Path,
    sample_frequency: float | None = TIMPEL_SAMPLE_FREQUENCY,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> dict[str, DataCollection]:
    """Load Timpel EIT data from path."""
    if not sample_frequency:
        sample_frequency = TIMPEL_SAMPLE_FREQUENCY

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
    time = np.arange(nframes + first_frame) / sample_frequency
    time = time[first_frame:]

    pixel_impedance = data[:, :1024]
    pixel_impedance = np.reshape(pixel_impedance, (-1, 32, 32), order="C")

    pixel_impedance = np.where(pixel_impedance == _NAN_VALUE, np.nan, pixel_impedance)

    eit_data = EITData(
        vendor=Vendor.TIMPEL,
        label="raw",
        path=path,
        nframes=nframes,
        time=time,
        sample_frequency=sample_frequency,
        pixel_impedance=pixel_impedance,
    )
    eitdata_collection = DataCollection(EITData, raw=eit_data)

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
            values=eit_data.calculate_global_impedance(),
            sample_frequency=sample_frequency,
        ),
    )
    continuousdata_collection.add(
        ContinuousData(
            label="airway_pressure_(timpel)",
            name="Airway pressure",
            unit="cmH2O",
            category="pressure",
            description="Airway pressure measured by Timpel device",
            time=time,
            values=data[:, 1024],
            sample_frequency=sample_frequency,
        ),
    )

    continuousdata_collection.add(
        ContinuousData(
            label="flow_(timpel)",
            name="Flow",
            unit="L/s",
            category="flow",
            description="Flow measures by Timpel device",
            time=time,
            values=data[:, 1025],
            sample_frequency=sample_frequency,
        ),
    )

    continuousdata_collection.add(
        ContinuousData(
            label="volume_(timpel)",
            name="Volume",
            unit="L",
            category="volume",
            description="Volume measured by Timpel device",
            time=time,
            values=data[:, 1026],
            sample_frequency=sample_frequency,
        ),
    )

    # extract sparse data
    sparsedata_collection = DataCollection(SparseData)

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

    gi = continuousdata_collection["global_impedance_(raw)"].values

    time_ranges, breaths = _make_breaths(time, min_indices, max_indices, gi)
    intervaldata_collection = DataCollection(IntervalData)
    intervaldata_collection.add(
        IntervalData(
            label="breaths_(timpel)",
            name="Breaths (Timpel)",
            unit=None,
            category="breaths",
            intervals=time_ranges,
            values=breaths,
            default_partial_inclusion=False,
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

    return {
        "eitdata_collection": eitdata_collection,
        "continuousdata_collection": continuousdata_collection,
        "sparsedata_collection": sparsedata_collection,
        "intervaldata_collection": intervaldata_collection,
    }


def _make_breaths(
    time: np.ndarray,
    min_indices: np.ndarray,
    max_indices: np.ndarray,
    gi: np.ndarray,
) -> tuple[list[tuple[float, float]], list[Breath]]:
    # TODO: replace section with BreathDetection._remove_doubles() and BreathDetection._remove_edge_cases() from
    # 41_breath_detection_psomhorst; this code was directly copied from b59ac54

    if len(min_indices) < 2 or len(max_indices) < 1:  # noqa: PLR2004
        return [], []

    valley_indices = min_indices.copy()
    peak_indices = max_indices.copy()

    keep_peaks = peak_indices > valley_indices[0]
    peak_indices = peak_indices[keep_peaks]

    keep_peaks = peak_indices < valley_indices[-1]
    peak_indices = peak_indices[keep_peaks]

    valley_values = gi[min_indices]
    peak_values = gi[max_indices]

    current_valley_index = 0
    while current_valley_index < len(valley_indices) - 1:
        start_index = valley_indices[current_valley_index]
        end_index = valley_indices[current_valley_index + 1]
        peaks_between_valleys = np.argwhere(
            (peak_indices > start_index) & (peak_indices < end_index),
        )
        if not len(peaks_between_valleys):
            # no peak between valleys, remove highest valley
            delete_valley_index = (
                current_valley_index
                if valley_values[current_valley_index] > valley_values[current_valley_index + 1]
                else current_valley_index + 1
            )
            valley_indices = np.delete(valley_indices, delete_valley_index)
            valley_values = np.delete(valley_values, delete_valley_index)
            continue

        if len(peaks_between_valleys) > 1:
            # multiple peaks between valleys, remove lowest peak
            delete_peak_index = (
                peaks_between_valleys[0]
                if peak_values[peaks_between_valleys[0]] < peak_values[peaks_between_valleys[1]]
                else peaks_between_valleys[1]
            )
            peak_indices = np.delete(peak_indices, delete_peak_index)
            peak_values = np.delete(peak_values, delete_peak_index)
            continue

        current_valley_index += 1

    breaths = []
    for start, end, middle in zip(valley_indices[:-1], valley_indices[1:], peak_indices, strict=True):
        breaths.append(((time[start], time[end]), Breath(time[start], time[middle], time[end])))

    time_ranges, values = zip(*breaths, strict=True)
    return list(time_ranges), list(values)
