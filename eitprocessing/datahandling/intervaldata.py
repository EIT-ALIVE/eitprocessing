from dataclasses import dataclass, field
from typing import Any, NamedTuple

from typing_extensions import Self


class TimeRange(NamedTuple):
    """A tuple containing the start time and end time of a time range."""

    start_time: float
    end_time: float


@dataclass
class IntervalData:
    """Container for single value data existing over a period of time."""

    label: str
    name: str
    unit: str | None
    category: str
    time_ranges: list[TimeRange | tuple[float, float]]
    values: list[Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    derived_from: list[Any] = field(default_factory=list)
    description: str = ""
    partial_inclusion: bool = False

    def __post_init__(self) -> None:
        self.time_ranges = [TimeRange._make(time_range) for time_range in self.time_ranges]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.label}')"

    def select_by_time(
        self,
        start: float | None = None,
        end: float | None = None,
        partial_inclusion: bool | None = None,
        label: str | None = None,
    ) -> Self:
        """Return only period data that overlaps (partly) with start and end time.

        Other types of data (e.g. ContinuousData and SparseData) support the start_inclusive and end_inclusive
        arguments. PeriodData does not. That means that selection by time of PeriodData probably works slightly
        different than other types of selecting/slicing data.
        """
        if partial_inclusion is None:
            partial_inclusion = self.partial_inclusion
        label = label or f"Sliced version of <{self.label}>"

        time_range_value_pairs = zip(self.time_ranges, self.values, strict=True)

        def keep_starting_on_or_before_end(item: tuple[TimeRange, Any]) -> bool:
            time_range, _ = item
            return time_range.start_time <= end

        def keep_ending_on_or_after_start(item: tuple[TimeRange, Any]) -> bool:
            time_range, _ = item
            return time_range.end_time >= start

        def keep_fully_overlapping(item: tuple[TimeRange, Any]) -> bool:
            time_range, _ = item
            if time_range.start_time < start:
                return False
            if time_range.end_time > end:
                return False
            return True

        time_range_value_pairs = filter(keep_starting_on_or_before_end, time_range_value_pairs)
        time_range_value_pairs = filter(keep_ending_on_or_after_start, time_range_value_pairs)

        if not partial_inclusion:
            time_range_value_pairs = filter(keep_fully_overlapping, time_range_value_pairs)

        time_ranges, values = zip(*time_range_value_pairs, strict=True)

        def replace_start_end_time(time_range: TimeRange) -> TimeRange:
            start_time_ = max(time_range.start_time, start)
            end_time_ = min(time_range.end_time, end)
            return TimeRange(start_time_, end_time_)

        time_ranges = list(map(replace_start_end_time, time_ranges))

        return self.__class__(
            label=label,
            name=self.name,
            unit=self.unit,
            category=self.category,
            derived_from=[*self.derived_from, self],
            time_ranges=time_ranges,
            values=values,
        )
