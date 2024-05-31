import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypeVar

import numpy as np
from typing_extensions import Self

from eitprocessing.datahandling.mixins.equality import Equivalence
from eitprocessing.datahandling.mixins.slicing import HasTimeIndexer

T = TypeVar("T", bound="IntervalData")


class TimeRange(NamedTuple):
    """A tuple containing the start time and end time of a time range."""

    start_time: float
    end_time: float


@dataclass(eq=False)
class IntervalData(Equivalence, HasTimeIndexer):
    """Container for interval data existing over a period of time.

    Interval data is data that constists for a given time interval. Examples are a ventilator setting (e.g.
    end-expiratory pressure), the position of a patient, a maneuver (end-expiratory hold) being performed, detected
    periods in the data, etc.

    Interval data consists of a number of time range-value pairs or time ranges without associated values. E.g. interval
    data with the label "expiratory_breath_hold" only requires time ranges for when expiratory breath holds were
    performed. Other interval data, e.g. "set_driving_pressure" do have associated values.

    Interval data can be selected by time through the `select_by_time(start_time, end_time)` method. Alternatively,
    `t[start_time:end_time]` can be used. When the start or end time overlaps with a time range, the time range and its
    associated value are included in the selection if `partial_inclusion` is `True`, but ignored if `partial_inclusion`
    is `False`. If the time range is partially included, the start and end times are trimmed to the start and end time
    of the selection.

    A potential use case where `partial_inclusion` should be set to `True` is "set_driving_pressure": you might want to
    keep the driving pressure that was set before the start of the selectioon. A use case where `partial_inclusion`
    should be set to `False` is "detected_breaths": you might want to ignore partial breaths that started before or
    ended after the selected period.

    Note that when selecting by time, the end time is included in the selection.

    Args:
        label: a computer-readable name
        name: a human-readable name
        unit: the unit associated with the data
        category: the category of data
        time_ranges: a list of time ranges (tuples containing a start time and end time)
        values: an optional list of values with the same length as time_ranges
        parameters: parameters used to derive the data
        derived_from: list of data sets this data was derived from
        description: extended human readible description of the data
        partial_inclusion: whether to include a trimmed version of a time range when selecting data
    """

    label: str = field(compare=False)
    name: str = field(compare=False)
    unit: str | None = field(metadata={"check_equivalence": True})
    category: str = field(metadata={"check_equivalence": True})
    time_ranges: list[TimeRange | tuple[float, float]]
    values: list[Any] | None = None
    parameters: dict[str, Any] = field(default_factory=dict, metadata={"check_equivalence": True})
    derived_from: list[Any] = field(default_factory=list, compare=False)
    description: str = field(compare=False, default="")
    partial_inclusion: bool = False

    def __post_init__(self) -> None:
        self.time_ranges = [TimeRange._make(time_range) for time_range in self.time_ranges]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.label}')"

    def __len__(self) -> int:
        return len(self.time_ranges)

    @property
    def has_values(self) -> bool:
        """True if the IntervalData has values, False otherwise."""
        return self.values is not None

    def select_by_time(  # noqa: C901
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        partial_inclusion: bool | None = None,
        newlabel: str | None = None,
    ) -> Self:
        """Return only period data that overlaps (partly) with start and end time.

        Other types of data (e.g. ContinuousData and SparseData) support the start_inclusive and end_inclusive
        arguments. PeriodData does not. That means that selection by time of PeriodData probably works slightly
        different than other types of selecting/slicing data.
        """
        if partial_inclusion is None:
            partial_inclusion = self.partial_inclusion
        newlabel = newlabel or self.label

        if start_time is None:
            start_time = self.time_ranges[0].start_time
        if end_time is None:
            end_time = self.time_ranges[-1].end_time

        def keep_starting_on_or_before_end(item: tuple[TimeRange, Any]) -> bool:
            time_range, _ = item
            return time_range.start_time <= end_time

        def keep_ending_on_or_after_start(item: tuple[TimeRange, Any]) -> bool:
            time_range, _ = item
            return time_range.end_time >= start_time

        def keep_fully_overlapping(item: tuple[TimeRange, Any]) -> bool:
            time_range, _ = item
            if time_range.start_time < start_time:
                return False
            if time_range.end_time > end_time:
                return False
            return True

        def replace_start_end_time(time_range: TimeRange) -> TimeRange:
            start_time_ = max(time_range.start_time, start_time)
            end_time_ = min(time_range.end_time, end_time)
            return TimeRange(start_time_, end_time_)

        iter_values = self.values or itertools.repeat(None)

        time_range_value_pairs: Any = zip(self.time_ranges, iter_values, strict=True)
        time_range_value_pairs = filter(keep_starting_on_or_before_end, time_range_value_pairs)
        time_range_value_pairs = list(filter(keep_ending_on_or_after_start, time_range_value_pairs))

        if not partial_inclusion:
            time_range_value_pairs = list(filter(keep_fully_overlapping, time_range_value_pairs))

        if len(time_range_value_pairs):
            time_ranges, values = zip(*time_range_value_pairs, strict=True)
            time_ranges = list(map(replace_start_end_time, time_ranges))
        else:
            time_ranges, values = [], []

        return self.__class__(
            label=newlabel,
            name=self.name,
            unit=self.unit,
            category=self.category,
            derived_from=[*self.derived_from, self],
            time_ranges=list(time_ranges),
            values=list(values) if self.has_values else None,
        )

    def concatenate(self: T, other: T, newlabel: str | None = None) -> T:  # noqa: D102, will be moved to mixin in future
        self.isequivalent(other, raise_=True)

        # TODO: make proper copy functions
        if not len(self):
            return copy.deepcopy(other)
        if not len(other):
            return copy.deepcopy(self)

        if other.time_ranges[0].start_time < self.time_ranges[-1].end_time:
            msg = f"{other} (b) starts before {self} (a) ends."
            raise ValueError(msg)

        cls = type(self)
        newlabel = newlabel or self.label

        if isinstance(self.values, list | tuple) and isinstance(other.values, list | tuple):
            new_values = self.values + other.values
        elif isinstance(self.values, np.ndarray) and isinstance(other.values, np.ndarray):
            new_values = np.concatenate((self.values, other.values))
        elif self.values is None and other.values is None:
            new_values = None
        else:
            msg = "self and other have different value types"
            raise TypeError(msg)

        return cls(
            label=newlabel,
            name=self.name,
            unit=self.unit,
            category=self.category,
            description=self.description,
            derived_from=[*self.derived_from, *other.derived_from, self, other],
            time_ranges=self.time_ranges + other.time_ranges,
            values=new_values,
        )
