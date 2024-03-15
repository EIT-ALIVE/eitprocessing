from functools import reduce
from pathlib import Path
from typing import NamedTuple

from eitprocessing.continuous_data import ContinuousData
from eitprocessing.data_collection import DataCollection
from eitprocessing.eit_data import EITData
from eitprocessing.eit_data.vendor import Vendor
from eitprocessing.sparse_data import SparseData


class LoadDataResult(NamedTuple):
    """Result from load_data(), containing three DataCollections with EITData, ContinuousData and SparseData."""

    eit_data: DataCollection[str, EITData]
    continuous_data: DataCollection[str, ContinuousData]
    sparse_data: DataCollection[str, SparseData]


def load_data(
    path: str | Path | list[str | Path],
    vendor: Vendor | str,
    framerate: float | None = None,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> tuple[DataCollection[EITData], DataCollection[ContinuousData], DataCollection[SparseData]]:
    """Load EIT data from path(s).

    Args:
        path: relative or absolute path(s) to data file.
        vendor: vendor indicating the device used.
        label: description of object for human interpretation.
            Defaults to "Sequence_<unique_id>".
        framerate: framerate at which the data was recorded.
            Default for Draeger: 20
            Default for Timpel: 50
            Default for Sentec: 50.2
        first_frame: index of first frame to load.
            Defaults to 0.
        max_frames: maximum number of frames to load.
            The actual number of frames can be lower than this if this
            would surpass the final frame.

    Raises:
        NotImplementedError: is raised when there is no loading method for
        the given vendor.

    Returns:
        LoadDataResult: a named tuple containing a DataCollection of EITData (result.eit_data or result[0]), a
        DataCollection of ContinuousData (result.continuous_data or result[1]) and a DataCollection of SparseData
        (result.sparse_data or result[2]).
    """
    from eitprocessing.continuous_data import ContinuousData
    from eitprocessing.data_collection import DataCollection
    from eitprocessing.eit_data.loading import draeger, sentec, timpel
    from eitprocessing.sparse_data import SparseData

    vendor = _ensure_vendor(vendor)
    load_func = {
        "draeger": draeger.load_from_single_path,
        "timpel": timpel.load_from_single_path,
        "sentec": sentec.load_from_single_path,
    }[vendor]

    first_frame = _check_first_frame(first_frame)

    paths = EITData.ensure_path_list(path)

    eit_datasets: list[DataCollection] = []
    continuous_datasets: list[DataCollection] = []
    sparse_datasets: list[DataCollection] = []

    for single_path in paths:
        single_path.resolve(strict=True)  # raise error if any file does not exist

    for single_path in paths:
        loaded_data = load_func(
            path=single_path,
            framerate=framerate,
            first_frame=first_frame,
            max_frames=max_frames,
        )

        if type(loaded_data) is not tuple:
            eit = loaded_data
            continuous = DataCollection(ContinuousData)
            sparse = DataCollection(SparseData)
        else:
            eit, continuous, sparse = loaded_data

        eit: DataCollection
        continuous: DataCollection
        sparse: DataCollection

        eit_datasets.append(eit)
        continuous_datasets.append(continuous)
        sparse_datasets.append(sparse)

    return LoadDataResult(
        reduce(DataCollection.concatenate, datasets)
        for datasets in (eit_datasets, continuous_datasets, sparse_datasets)
    )


def _check_first_frame(first_frame: int | None) -> int:
    if first_frame is None:
        first_frame = 0
    if int(first_frame) != first_frame:
        msg = f"`first_frame` must be an int, but was given as {first_frame} (type: {type(first_frame)})"
        raise TypeError(msg)
    if first_frame < 0:
        msg = f"`first_frame` can not be negative, but was given as {first_frame}"
        raise ValueError(msg)
    return int(first_frame)


def _ensure_vendor(vendor: Vendor | str) -> Vendor:
    """Check whether vendor exists, and assure it's a Vendor object."""
    try:
        return Vendor(vendor)
    except ValueError as e:
        msg = f"Unknown vendor {vendor}."
        raise ValueError(msg) from e
