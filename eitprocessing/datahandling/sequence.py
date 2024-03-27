from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.mixins.equality import Equivalence
from eitprocessing.datahandling.mixins.slicing import SelectByIndex, TimeIndexer
from eitprocessing.datahandling.sparsedata import SparseData

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


@dataclass(eq=False)
class Sequence(Equivalence, SelectByIndex):
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
    """

    label: str | None = None
    name: str | None = None
    description: str = ""
    eit_data: DataCollection = field(default_factory=lambda: DataCollection(EITData))
    continuous_data: DataCollection = field(default_factory=lambda: DataCollection(ContinuousData))
    sparse_data: DataCollection = field(default_factory=lambda: DataCollection(SparseData))

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

    @classmethod
    def concatenate(
        cls,
        a: Sequence,
        b: Sequence,
    ) -> Sequence:
        """Create a merge of two Sequence objects."""
        # TODO: rewrite

        eit_data = a.eit_data.concatenate(b.eit_data) if a.eit_data and b.eit_data else None

        # TODO: add concatenation of other attached objects

        return a.__class__(eit_data=eit_data)

    def _sliced_copy(self, start_index: int, end_index: int, label: str) -> Self:
        eit_data = DataCollection(EITData)
        for value in self.eit_data.values():
            eit_data.add(value[start_index:end_index])

        continuous_data = DataCollection(ContinuousData)
        for value in self.continuous_data.values():
            continuous_data.add(value[start_index:end_index])

        sparse_data = DataCollection(SparseData)
        if start_index >= len(self.time):
            msg = "start_index larger than length of time axis"
            raise ValueError(msg)

        time = self.time[start_index:end_index]
        for value in self.sparse_data.values():
            sparse_data.add(value.t[time[0], time[-1]])

        return self.__class__(
            label=label,
            name=f"Sliced copy of <{self.name}>",
            description=f"Sliced copy of <{self.description}>",
            eit_data=eit_data,
            continuous_data=continuous_data,
            sparse_data=sparse_data,
        )

    def select_by_time(
        self,
        start: float | None = None,
        end: float | None = None,
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
            eit_data=self.eit_data.select_by_time(start, end, start_inclusive, end_inclusive),
            continuous_data=self.continuous_data.select_by_time(start, end, start_inclusive, end_inclusive),
            sparse_data=self.sparse_data.select_by_time(start, end, start_inclusive, end_inclusive),
        )

    @property
    def t(self):
        """Time indexer.

        See slicing.TimeIndexer.
        """
        return TimeIndexer(self)
