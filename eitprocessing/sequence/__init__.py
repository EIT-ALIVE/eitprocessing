from __future__ import annotations
import bisect
import copy
import warnings
from dataclasses import dataclass
import numpy as np
from typing_extensions import Self
from eitprocessing.eit_data import EITData
from eitprocessing.mixins.equality import Equivalence
from eitprocessing.mixins.equality import EquivalenceError


@dataclass(eq=False)
class Sequence(Equivalence):
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
    eit_data: EITData | None = None

    def __post_init__(self):
        if self.label is None:
            self.label = f"Sequence_{id(self)}"

    def isequivalent(
        self,
        other: Self,
        raise_: bool = False,
    ) -> bool:
        # fmt: off
        checks = {
            "Only one of the sequences contains EIT data.": bool(self.eit_data) is bool(other.eit_data),  # both True or both False
            "EITData is not equivalent.": EITData.isequivalent(self.eit_data, other.eit_data, raise_),
            # TODO: add other attached objects for equivalence
        }
        # fmt: on
        return super().isequivalent(other, raise_, checks)

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
        try:
            Sequence.isequivalent(a, b, raise_=True)
        except EquivalenceError as e:
            raise type(e)(f"Sequences could not be merged: {e}") from e

        if a.eit_data and b.eit_data:
            eit_data = EITData.concatenate(a.eit_data, b.eit_data)
        else:
            eit_data = None

        # TODO: add concatenation of other attached objects

        label = label or f"Concatenation of <{a.label}> and <{b.label}>"

        return a.__class__(label=label, eit_data=eit_data)

    def select_by_index(
        self,
        indices: slice,
        label: str | None = None,
    ):
        ...
        # TODO: rewrite to use EITData, SparseData and ContinuousData

    def __getitem__(self, indices: slice):
        # TODO: reconsider API
        return self.select_by_index(indices)

    def select_by_time(  # pylint: disable=too-many-arguments
        self,
        start: float | int | None = None,
        end: float | int | None = None,
        start_inclusive: bool = True,
        end_inclusive: bool = False,
        label: str | None = None,
    ) -> Sequence:
        """Select subset of sequence by the `Sequence.time` information (i.e.
        based on the time stamp).

        Args:
            start (float | int | None, optional): starting time point.
                Defaults to None.
            end (float | int | None, optional): ending time point.
                Defaults to None.
            start_inclusive (bool, optional): include starting timepoint if
                `start` is present in `Sequence.time`.
                Defaults to True.
            end_inclusive (bool, optional): include ending timepoint if
                `end` is present in `Sequence.time`.
                Defaults to False.

        Raises:
            ValueError: if the Sequence.time is not sorted

        Returns:
            Sequence: a slice of `self` based on time information given.
        """

        # TODO: rewrite

        if not any((start, end)):
            warnings.warn("No starting or end timepoint was selected.")
            return self
        if not np.all(np.sort(self.time) == self.time):
            raise ValueError(
                f"Time stamps for {self} are not sorted and therefor data"
                "cannot be selected by time."
            )

        if start is None:
            start_index = 0
        elif start_inclusive:
            start_index = bisect.bisect_left(self.time, start)
        else:
            start_index = bisect.bisect_right(self.time, start)

        if end is None:
            end_index = len(self)
        elif end_inclusive:
            end_index = bisect.bisect_right(self.time, end) - 1
        else:
            end_index = bisect.bisect_left(self.time, end) - 1

        return self.select_by_index(slice(start_index, end_index), label=label)

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
