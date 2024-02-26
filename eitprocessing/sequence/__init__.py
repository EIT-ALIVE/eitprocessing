from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from eitprocessing.continuous_data import ContinuousData
from eitprocessing.data_collection import DataCollection
from eitprocessing.eit_data import EITData
from eitprocessing.mixins.addition import Addition
from eitprocessing.mixins.slicing import SelectByTime
from eitprocessing.sparse_data import SparseData

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


@dataclass(eq=False)
class Sequence(Addition, SelectByTime):
    """Sequence of timepoints containing EIT and/or waveform data.

    A Sequence is a representation of a continuous set of data points, either EIT frames,
    waveform data, or both. A Sequence can consist of an entire measurement, a section of a
    measurement, a single breath, or even a portion of a breath.
    A sequence can be split up into separate sections of a measurement or multiple (similar)
    Sequence objects can be merged together to form a single Sequence.

    EIT data is contained within Framesets. A Frameset shares the time axis with a Sequence.

    Args:
        framesets (dict[str, Frameset]): dictionary of framesets
        events (list[Event]): list of Event objects in data
        timing_errors (list[TimingError]): list of TimingError objects in data
        phases (list[PhaseIndicator]): list of PhaseIndicator objects in data
    """

    eit_data: DataCollection = field(default_factory=lambda: DataCollection(EITData))
    continuous_data: DataCollection = field(default_factory=lambda: DataCollection(ContinuousData))
    sparse_data: DataCollection = field(default_factory=lambda: DataCollection(SparseData))

    def __post_init__(self):
        self._check_equivalence = []

    @property
    def time(self) -> np.ndarray:
        if len(self.eit_data):
            return self.eit_data["raw"].time
        if len(self.continuous_data):
            return self.continuous_data.values()[0].time

        msg = "Sequence has no timed data"
        raise AttributeError(msg)

    def __len__(self):
        return len(self.time)

    def _sliced_copy(self, start_index: int, end_index: int) -> Self:
        eit_data = DataCollection(EITData)
        for key, value in self.eit_data.items():
            eit_data.add(value[start_index:end_index])

        continuous_data = DataCollection(ContinuousData)
        for key, value in self.continuous_data.items():
            continuous_data.add(value[start_index:end_index])

        sparse_data = DataCollection(SparseData)
        if start_index >= len(self.time):
            msg = "start_index larger than length of time axis"
            raise ValueError(msg)

        time = self.time[start_index:end_index]
        for value in self.sparse_data.values():
            sparse_data.add(value.t[time[0], time[-1]])

        return self.__class__(
            eit_data=eit_data,
            continuous_data=continuous_data,
            sparse_data=sparse_data,
        )
