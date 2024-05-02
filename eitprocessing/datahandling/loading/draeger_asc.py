# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.datahandling.sparsedata import SparseData

# %%

column_unit_pattern = re.compile(r"^(?P<continuous>~?)(?P<quantity>.+?)(\ \[(?P<unit>.+?)\])?$")

# %%
path = Path("tests/test_data/Draeger_Test_4.asc")


# %%
def load_draeger_asc_file(path: Path):
    start_of_table = _find_start_of_data(path)
    data = pd.read_csv(
        path,
        skiprows=start_of_table,
        delimiter="\t",
        decimal=",",
        encoding="latin1",
        na_values="-",
        index_col=False,
    )
    data["Time"] = data["Time"] * 24 * 60 * 60

    time = data["Time"].to_numpy()
    data = data.drop(columns=["Time", "Image", "Timing Error"])

    continuous_data_collection = DataCollection(ContinuousData)
    sparse_data_collection = DataCollection(SparseData)
    interval_data_collection = DataCollection(IntervalData)

    data = _parse_minmax_values(data, time, sparse_data_collection)
    data = _parse_events(data, time, sparse_data_collection)

    for column_name, column_data in data.items():
        if not (match := column_unit_pattern.match(column_name)):
            msg = f"Could not parse column name {column_name}"
            raise ValueError(msg)

        unit = match.group("unit")
        quantity = match.group("quantity")
        is_continuous = bool(match.group("continuous"))
        is_interval = False

        if quantity.startswith(("Local ",)) or quantity == "Global":
            is_continuous = True
            unit = "a.u."

        continuous_data = ContinuousData(
            label=quantity,
            name=quantity.capitalize(),
            description="",
            unit=unit,
            time=time,
            values=column_data.to_numpy(),
            category="",
        )
        if is_continuous:
            continuous_data_collection.add(continuous_data)
        elif is_interval:
            interval_data_collection.add(continuous_data.to_intervaldata())
        else:
            sparse_data_collection.add(continuous_data.to_sparsedata())

    return Sequence(
        continuous_data=continuous_data_collection,
        sparse_data=sparse_data_collection,
        interval_data=interval_data_collection,
    ), data


# %%
def _find_start_of_data(path: Path) -> int:
    """Find where the Medibus data starts in the file."""
    with path.open("rb") as fh:
        skip_lines = 0
        while line := fh.readline():
            if line.startswith(b"Image") and b"MinMax" in line:
                break

            skip_lines += 1

        return skip_lines


def _parse_minmax_values(
    data: pd.DataFrame,
    time: np.ndarray,
    sparse_data_collection: DataCollection[SparseData],
) -> None:
    min_values = np.flatnonzero(data["MinMax"] == -1)
    max_values = np.flatnonzero(data["MinMax"] == 1)

    sparse_data_collection.add(
        SparseData(
            label="min_values_(draeger)",
            name="Location of minimum values",
            description="",
            unit=None,
            time=time[min_values],
            category="minvalues",
        ),
        SparseData(
            label="max_values_(draeger)",
            name="Location of maximum values",
            description="",
            unit=None,
            time=time[max_values],
            category="maxvalues",
        ),
    )

    return data.drop(columns="MinMax")


def _parse_events(data: pd.DataFrame, time: np.ndarray, sparse_data_collection: DataCollection[SparseData]) -> None:
    event_indices = np.flatnonzero(data["Event"].diff() == 1)
    event_texts = data.loc[event_indices, "EventText"].str.strip().to_list()
    event_times = time[event_indices]

    sparse_data_collection.add(
        SparseData(
            label="events_(draeger)",
            name="Events",
            description="",
            unit=None,
            time=event_times,
            category="events",
            values=event_texts,
        ),
    )

    return data.drop(columns=["Event", "EventText"])


# %%

# %%
seq, data = load_draeger_asc_file(path)
