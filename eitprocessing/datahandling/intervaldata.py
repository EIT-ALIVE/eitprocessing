import copy
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypeVar

import numpy as np
from typing_extensions import Self

from eitprocessing.datahandling import DataContainer
from eitprocessing.datahandling.mixins.slicing import HasTimeIndexer, SelectByIndex

T = TypeVar("T", bound="IntervalData")


class Interval(NamedTuple):
    """A tuple containing the start time and end time of an interval."""

    start_time: float
    end_time: float


@dataclass(eq=False)
class IntervalData(DataContainer, SelectByIndex, HasTimeIndexer):
    """Container for interval data existing over a period of time.

    Interval data is data that consists for a given time interval. Examples are a ventilator setting (e.g.
    end-expiratory pressure), the position of a patient, a maneuver (end-expiratory hold) being performed, detected
    periods in the data, etc.

    Interval data consists of a number of intervals that may or may not have values associated with them.

    Examples of IntervalData with associated values are certain ventilator settings (e.g. end-expiratory pressure) and
    the position of a patient. Examples of IntervalData without associated values are indicators of maneouvres (e.g. a
    breath hold) or detected occurences (e.g. a breath).


    Interval data can be selected by time through the `select_by_time(start_time, end_time)` method. Alternatively,
    `t[start_time:end_time]` can be used.

    Args:
        label: Computer readable label identifying this dataset.
        name: Human readable name for the data.
        unit: The unit of the data, if applicable.
        category: Category the data falls into, e.g. 'breath'.
        intervals: A list of intervals (tuples containing a start time and end time).
        values: An optional list of values associated with each interval.
        parameters: Parameters used to derive the data.
        derived_from: Traceback of intermediates from which the current data was derived.
        description: Extended human readible description of the data.
        default_partial_inclusion: Whether to include a trimmed version of an interval when selecting data
    """

    label: str = field(compare=False)
    name: str = field(compare=False, repr=False)
    unit: str | None = field(metadata={"check_equivalence": True}, repr=False)
    category: str = field(metadata={"check_equivalence": True}, repr=False)
    intervals: list[Interval | tuple[float, float]] = field(repr=False)
    values: list[Any] | None = field(repr=False, default=None)
    parameters: dict[str, Any] = field(default_factory=dict, metadata={"check_equivalence": True}, repr=False)
    derived_from: list[Any] = field(default_factory=list, compare=False, repr=False)
    description: str = field(compare=False, default="", repr=False)
    default_partial_inclusion: bool = field(repr=False, default=False)

    def __post_init__(self) -> None:
        self.intervals = [Interval._make(interval) for interval in self.intervals]

        if self.has_values and (lv := len(self.values)) != (lt := len(self.intervals)):
            msg = f"The number of time points ({lt}) does not match the number of values ({lv})."
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.intervals)

    @property
    def has_values(self) -> bool:
        """True if the IntervalData has values, False otherwise."""
        return self.values is not None

    def _sliced_copy(
        self,
        start_index: int,
        end_index: int,
        newlabel: str,
    ) -> Self:
        cls = type(self)
        intervals = self.intervals[start_index:end_index]
        values = self.values[start_index:end_index] if self.has_values else None
        description = f"Slice ({start_index}-{end_index}) of <{self.description}>"

        return cls(
            label=newlabel,
            name=self.name,
            unit=self.unit,
            category=self.category,
            description=description,
            derived_from=[*self.derived_from, self],
            intervals=intervals,
            values=values,
        )

    def select_by_time(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        partial_inclusion: bool | None = None,
        newlabel: str | None = None,
    ) -> Self:
        """Create a new copy of the object, selecting data between start_time and end_time.

        This function returns a shortened copy of the object, containing data from the specified start_time to end_time.

        If `partial_inclusion` is set to `True`, any intervals that overlap with the start_time or end_time are included
        in the selection, and their times are adjusted to fit within the specified range. If `partial_inclusion` is
        `False`, intervals that overlap the start or end times are excluded from the selection.

        For example:
        - Set `partial_inclusion` to `True` for cases like "set_driving_pressure" where you want to include settings
        that were active before the start_time.
        - Set `partial_inclusion` to `False` for cases like "detected_breaths" where you want to exclude partial data
        that doesn't fully fit within the time range.

        Note that the end_time is always included in the selection if it is present in the original object.

        Args:
            start_time: The earliest time point to include in the copy.
            end_time: The latest time point to include in the copy.
            partial_inclusion: Whether to include intervals that overlap with the start_time or end_time.
            newlabel: A new label for the copied object.
        """
        newlabel = newlabel or self.label

        if start_time is None and end_time is None:
            copy_ = copy.deepcopy(self)
            copy_.derived_from.append(self)
            copy_.label = newlabel
            return copy_

        partial_inclusion = partial_inclusion or self.default_partial_inclusion

        selection_start = start_time or self.intervals[0].start_time
        selection_end = end_time or self.intervals[-1].end_time

        numbered_filtered_intervals = [
            (i, self._replace_start_end_time(interval, selection_start, selection_end))
            for i, interval in enumerate(self.intervals)
            if self._keep_overlapping(interval, selection_start, selection_end, partial_inclusion)
        ]

        try:
            indices, filtered_intervals = zip(*numbered_filtered_intervals, strict=True)
            values = [self.values[i] for i in indices] if self.has_values else None
        except ValueError:
            filtered_intervals = []
            values = [] if self.has_values else None

        return type(self)(
            label=newlabel,
            name=self.name,
            unit=self.unit,
            category=self.category,
            derived_from=[*self.derived_from, self],
            intervals=list(filtered_intervals),
            values=values,
        )

    @staticmethod
    def _keep_overlapping(
        interval: Interval,
        selection_start: float,
        selection_end: float,
        keep_partial_overlapping: bool,
    ) -> bool:
        """Helper function for filtering overlapping interval-value pairs."""
        if keep_partial_overlapping:
            return interval.start_time < selection_end and interval.end_time > selection_start

        return interval.start_time >= selection_start and interval.end_time <= selection_end

    @staticmethod
    def _replace_start_end_time(
        interval: Interval,
        selection_start: float,
        selection_end: float,
    ) -> Interval:
        """Helper function to replace start and end time after filtering interval-value pairs."""
        start_time_ = max(interval.start_time, selection_start)
        end_time_ = min(interval.end_time, selection_end)
        return Interval(start_time_, end_time_)

    def __add__(self: Self, other: Self) -> Self:
        return self.concatenate(other)

    def concatenate(self: Self, other: Self, newlabel: str | None = None) -> Self:  # noqa: D102, will be moved to mixin in future
        self.isequivalent(other, raise_=True)

        # TODO: make proper copy functions
        if not len(self):
            return copy.deepcopy(other)
        if not len(other):
            return copy.deepcopy(self)

        if other.intervals[0].start_time < self.intervals[-1].end_time:
            msg = (
                "Concatenation failed. "
                f"Second dataset ({other.name}) may not start before the first ({self.name}) ends."
            )
            raise ValueError(msg)

        cls = type(self)
        newlabel = newlabel or self.label

        if type(self.values) is not type(other.values):
            msg = "Concatenation failed because value types are non-identical."
            raise TypeError(msg)
        if not self.has_values:
            new_values = None
        elif isinstance(self.values, np.ndarray):
            new_values = np.concatenate((self.values, other.values))
        else:
            try:
                new_values = self.values + other.values
            except TypeError as e:
                msg = "Concatenation failed because values could not be concatenated."
                raise TypeError(msg) from e

        return cls(
            label=newlabel,
            name=self.name,
            unit=self.unit,
            category=self.category,
            description=self.description,
            derived_from=[*self.derived_from, *other.derived_from, self, other],
            intervals=self.intervals + other.intervals,
            values=new_values,
        )
