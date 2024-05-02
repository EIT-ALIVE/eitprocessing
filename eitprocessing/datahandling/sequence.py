from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.mixins.equality import Equivalence
from eitprocessing.datahandling.mixins.slicing import HasTimeIndexer, SelectByTime
from eitprocessing.datahandling.sparsedata import SparseData

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


@dataclass(eq=False)
class Sequence(Equivalence, SelectByTime, HasTimeIndexer):
    """Sequence of timepoints containing respiratory data.

    A Sequence object is a representation of data points over time. These data can consist of any combination of EIT
    frames (`EITData`), waveform data (`ContinuousData`) from different sources, or individual events (`SparseData`)
    occurring at any given timepoint.
    A Sequence can consist of an entire measurement, a section of a measurement, a single breath, or even a portion of a
    breath. A Sequence can consist of multiple sets of each type of data from the same time-points or can be a single
    measurement from just one source.

    A Sequence can be split up into separate sections of a measurement or multiple (similar) Sequence objects can be
    merged together to form a single Sequence.

    Args:
        label: Computer readable naming of the instance.
        name: Human readable naming of the instance.
        description: Human readable extended description of the data.
        eit_data: Collection of one or more sets of EIT data frames.
        continuous_data: Collection of one or more sets of continuous data points.
        sparse_data: Collection of one or more sets of individual data points.
    """  # TODO: check that docstring is up to date

    label: str | None = None
    name: str | None = None
    description: str = ""
    eit_data: DataCollection = field(default_factory=lambda: DataCollection(EITData), repr=False)
    continuous_data: DataCollection = field(default_factory=lambda: DataCollection(ContinuousData), repr=False)
    sparse_data: DataCollection = field(default_factory=lambda: DataCollection(SparseData), repr=False)

    def __post_init__(self):
        if not self.label:
            self.label = f"Sequence_{id(self)}"
        self.name = self.name or self.label

    @property
    def time(self) -> np.ndarray:
        """Time axis from either EITData or ContinuousData."""
        if len(self.eit_data):
            return self.eit_data["raw"].time
        if len(self.continuous_data):
            return next(self.continuous_data.values())

        msg = "Sequence has no timed data"
        raise AttributeError(msg)

    def __len__(self):
        return len(self.time)

    def __add__(self, other: Sequence) -> Sequence:
        return self.concatenate(self, other)

    @classmethod  # TODO: why is this a class method? In other cases it's instance method
    def concatenate(
        cls,
        a: Sequence,
        b: Sequence,
        newlabel: str | None = None,
    ) -> Sequence:
        """Create a merge of two Sequence objects."""
        # TODO: rewrite

        concat_eit = a.eit_data.concatenate(b.eit_data)
        concat_continuous = a.continuous_data.concatenate(b.continuous_data)
        concat_sparse = a.sparse_data.concatenate(b.sparse_data)

        newlabel = newlabel or f"Merge of <{a.label}> and <{b.label}>"
        # TODO: add concatenation of other attached objects

        return a.__class__(
            eit_data=concat_eit,
            continuous_data=concat_continuous,
            sparse_data=concat_sparse,
            label=newlabel,
        )

    def _sliced_copy(self, start_index: int, end_index: int, newlabel: str) -> Self:  # noqa: ARG002
        if start_index >= len(self.time):
            msg = "start_index larger than length of time axis"
            raise ValueError(msg)
        time = self.time[start_index:end_index]

        sliced_eit = DataCollection(EITData)
        for value in self.eit_data.values():
            sliced_eit.add(value[start_index:end_index])

        sliced_continuous = DataCollection(ContinuousData)
        for value in self.continuous_data.values():
            sliced_continuous.add(value[start_index:end_index])

        sliced_sparse = DataCollection(SparseData)
        for value in self.sparse_data.values():
            sliced_sparse.add(value.t[time[0], time[-1]])

        return self.__class__(
            label=self.label,  # newlabel gives errors
            name=f"Sliced copy of <{self.name}>",
            description=f"Sliced copy of <{self.description}>",
            eit_data=sliced_eit,
            continuous_data=sliced_continuous,
            sparse_data=sliced_sparse,
        )

    def select_by_time(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        start_inclusive: bool = True,
        end_inclusive: bool = False,
        label: str | None = None,
        name: str | None = None,
        description: str | None = "",
    ) -> Self:
        """Return a sliced version of the Sequence.

        See SelectByTime.select_by_time().
        """
        if not label:
            label = f"copy_of_<{self.label}>"
        if not name:
            f"Sliced copy of <{self.name}>"

        return self.__class__(
            label=label,
            name=name,
            description=description,
            # perform select_by_time() on all four data types
            **{
                key: getattr(self, key).select_by_time(
                    start_time=start_time,
                    end_time=end_time,
                    start_inclusive=start_inclusive,
                    end_inclusive=end_inclusive,
                )
                for key in ("eit_data", "continuous_data", "sparse_data")
            },
        )
