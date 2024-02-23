from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from eitprocessing.continuous_data import ContinuousData
from eitprocessing.data_collection import DataCollection
from eitprocessing.eit_data import EITData
from eitprocessing.mixins.equality import Equivalence
from eitprocessing.mixins.slicing import SelectByTime
from eitprocessing.sparse_data import SparseData

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass(eq=False)
class Sequence(Equivalence, SelectByTime):
    """Sequence of timepoints containing EIT and/or waveform data.

    A Sequence is a representation of a continuous set of data points, either EIT frames,
    waveform data, or both. A Sequence can consist of an entire measurement, a section of a
    measurement, a single breath, or even a portion of a breath.
    A sequence can be split up into separate sections of a measurement or multiple (similar)
    Sequence objects can be merged together to form a single Sequence.

    EIT data is contained within Framesets. A Frameset shares the time axis with a Sequence.

    Args:
        label (str): description of object for human interpretation.
            Defaults to "Sequence_<unique_id>".
        framesets (dict[str, Frameset]): dictionary of framesets
        events (list[Event]): list of Event objects in data
        timing_errors (list[TimingError]): list of TimingError objects in data
        phases (list[PhaseIndicator]): list of PhaseIndicator objects in data
    """

    label: str | None = None
    eit_data: DataCollection = field(default_factory=DataCollection(EITData))
    continuous_data: DataCollection = field(default_factory=DataCollection(ContinuousData))
    sparse_data: DataCollection = field(default_factory=DataCollection(SparseData))

    def __post_init__(self):
        if self.label is None:
            self.label = f"Sequence_{id(self)}"

    def __add__(self, other: Sequence) -> Sequence:
        return self.concatenate(self, other)

    @classmethod
    def concatenate(
        cls,
        a: Sequence,
        b: Sequence,
        label: str | None = None,
    ) -> Sequence:
        """Create a merge of two Sequence objects."""
        # TODO: rewrite

        eit_data = EITData.concatenate(a.eit_data, b.eit_data) if a.eit_data and b.eit_data else None

        # TODO: add concatenation of other attached objects

        label = label or f"Concatenation of <{a.label}> and <{b.label}>"

        return a.__class__(label=label, eit_data=eit_data)

    def _sliced_copy(self, start_index: int, end_index: int, label: str) -> Self:
        eit_data = DataCollection(EITData)
        for key, value in self.eit_data:
            eit_data.add(key, value[start_index:end_index])

        continuous_data = DataCollection(ContinuousData)
        for key, value in self.continuous_data:
            continuous_data.add(key, value[start_index:end_index])

        sparse_data = DataCollection(SparseData)
        start_time = self.time[start_index]
        end_time = self.time[end_index]
        for key, value in self.sparse_data:
            sparse_data.add(key, value.t[start_time:end_time])

        return self.__class__(
            label,
            eit_data=eit_data,
            continuous_data=continuous_data,
            sparse_data=sparse_data,
        )

    def deepcopy(
        self,
        label: str | None = None,
        relabel: bool | None = True,
    ) -> Sequence:
        """Create a deep copy of `Sequence` object.

        Args:
            label (str): Create a new `label` for the copy.
                Defaults to None, which will trigger behavior described for relabel (below)
            relabel (bool): If `True` (default), the label of self is re-used for the copy,
                otherwise the following label is assigned f"Deepcopy of {self.label}".
                Note that this setting is ignored if a label is given.

        Returns:
            Sequence: a deep copy of self
        """
        # TODO: rewrite for efficiency

        obj = copy.deepcopy(self)
        if label:
            obj.label = label
        elif relabel:
            obj.label = f"Copy of <{self.label}>"
        return obj
