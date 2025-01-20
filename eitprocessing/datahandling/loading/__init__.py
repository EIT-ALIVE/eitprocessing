from functools import reduce
from pathlib import Path

from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.sequence import Sequence


def load_eit_data(
    path: str | Path | list[str | Path],
    vendor: Vendor | str,
    label: str | None = None,
    name: str | None = None,
    description: str = "",
    sample_frequency: float | None = None,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> Sequence:
    """Load EIT data from path(s).

    Args:
        path: relative or absolute path(s) to data file.
        vendor: vendor indicating the device used.
            Note: for load functions of specific vendors (e.g. `load_draeger_data`), this argument is defaulted to the
            correct vendor.
        label: short description of sequence for computer interpretation.
            Defaults to "Sequence_<unique_id>".
        name: short description of sequence for human interpretation.
            Defaults to the same value as label.
        description: long description of sequence for human interpretation.
        sample_frequency: sample frequency at which the data was recorded.
            No default for Draeger. Will be autodetected. Warns if autodetected differs from provided.
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
        Sequence: a Sequence with the given label, name and description, containing the loaded data.

    Example:
    ```
    >>> sequence = load_eit_data(
    ...     ["path/to/file1", "path/to/file2"],
    ...     vendor="sentec",
    ...     label="initial_measurement"
    ... )
    >>> pixel_impedance = sequence.eit_data["raw"].pixel_impedance
    ```
    """
    from eitprocessing.datahandling.loading import draeger, sentec, timpel  # not in top level to avoid circular import

    vendor = _ensure_vendor(vendor)
    load_from_single_path = {
        Vendor.DRAEGER: draeger.load_from_single_path,
        Vendor.TIMPEL: timpel.load_from_single_path,
        Vendor.SENTEC: sentec.load_from_single_path,
    }[vendor]

    first_frame = _check_first_frame(first_frame)

    paths = EITData.ensure_path_list(path)

    eit_datasets: list[DataCollection] = []
    continuous_datasets: list[DataCollection] = []
    sparse_datasets: list[DataCollection] = []
    interval_datasets: list[DataCollection] = []

    for single_path in paths:
        single_path.resolve(strict=True)  # raise error if any file does not exist

    for single_path in paths:
        loaded_data = load_from_single_path(
            path=single_path,
            sample_frequency=sample_frequency,
            first_frame=first_frame,
            max_frames=max_frames,
        )

        eit_datasets.append(loaded_data["eitdata_collection"])
        continuous_datasets.append(loaded_data["continuousdata_collection"])
        sparse_datasets.append(loaded_data["sparsedata_collection"])
        interval_datasets.append(loaded_data["intervaldata_collection"])

    return Sequence(
        label=label,
        name=name,
        description=description,
        eit_data=reduce(DataCollection.concatenate, eit_datasets),
        continuous_data=reduce(DataCollection.concatenate, continuous_datasets),
        sparse_data=reduce(DataCollection.concatenate, sparse_datasets),
        interval_data=reduce(DataCollection.concatenate, interval_datasets),
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
    """Check whether loading method for vendor exists, and ensure it's a Vendor object."""
    try:
        return Vendor(vendor)
    except ValueError as e:
        msg = f"No loading method for {vendor} exists."
        raise NotImplementedError(msg) from e
