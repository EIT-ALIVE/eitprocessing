from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self

# TODO: when slice by time is implemented, remove line below to activate linting
# ruff: noqa


class SelectByIndex(ABC):
    """Adds slicing functionality to subclass by implementing `__getitem__`.

    Subclasses must implement a `_sliced_copy` function that defines what should
    happen when the object is sliced. This class ensures that when calling a
    slice between square brackets (as e.g. done for lists) then return the
    expected sliced object.
    """

    label: str

    def __getitem__(self, key: slice | int):
        if isinstance(key, slice):
            if key.step and key.step != 1:
                msg = f"Can't slice {self.__class__} object with steps other than 1."
                raise ValueError(msg)
            start_index = key.start
            end_index = key.stop
            return self.select_by_index(start_index, end_index)

        if isinstance(key, int):
            return self.select_by_index(start=key, end=key + 1)

        msg = f"Invalid slicing input. Should be `slice` or `int`, not {type(key)}."
        raise TypeError(msg)

    def select_by_index(
        self,
        start: int | None = None,
        end: int | None = None,
        newlabel: str | None = None,
    ) -> Self:
        """De facto implementation of the `__getitem__ function.

        This function can also be called directly to add a label to the sliced
        object. Otherwise a default label describing the slice and original
        object is attached.
        """
        if start is None and end is None:
            warnings.warn("No starting or end timepoint was selected.")
            return self

        start = start or 0
        end = end or len(self)
        newlabel = newlabel or f"Slice ({start}-{end}] of <{self.label}>"

        return self._sliced_copy(start_index=start, end_index=end, newlabel=newlabel)

    @abstractmethod
    def _sliced_copy(
        self,
        start_index: int,
        end_index: int,
        newlabel: str,
    ) -> Self:
        """Slicing method that must be implemented by all subclasses.

        Must return a copy of self object with all attached data within selected
        indices.
        """
        ...


class HasTimeIndexer:
    """Gives access to a TimeIndexer object that can be used to slice by time."""

    @property
    def t(self) -> TimeIndexer:
        """Slicing an object using the time axis instead of indices.

        Example:
        ```
        >>> sequence = load_eit_data(<path>, ...)
        >>> time_slice1 = sequence.t[tp_start:tp_end]
        >>> time_slice2 = sequence.select_by_time(tp_start, tp_end)
        >>> time_slice1 == time_slice2
        True
        ```
        """
        return TimeIndexer(self)


class SelectByTime(SelectByIndex, HasTimeIndexer):
    """Adds methods for slicing by time rather than index."""

    def select_by_time(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        n_before: int = 0,
        n_after: int = 0,
        # start_behavior: Literal["force_inclusive", "inclusive", "exclusive"] = "inclusive",
        # end_behavior: Literal["force_exclusive", "exclusive", "inclusive"] = "exclusive",
        label: str | None = None,
    ) -> Self:
        """Slice object according to time stamps (i.e. its value, not its index).

        The sliced object must contain a time axis.

        Args:
            start_time: Start time of new object (inclusive). Unless it is lower than the first timepoint, `start_time`
                will be present in the time axis of the sliced object. This means that the first timepoint of the sliced
                object will be equal to `start_time` if it exists on the unsliced object, and otherwise the last
                timepoint preceding `start_time`.
                Defaults to None, which is the first time point of the object.
            end_time: End time of new object (exclusive). The final timepoint of the sliced object will be the last
                timepoint preceding `end_time`, irrespective of whether it exists in the unsliced object.
            n_before: Additional time points to include (or exclude if negative) before first time point defined above.
            n_after: Additional time points to include (or exclude if negative) after last time point defined above.

        Raises:
            TypeError: if `self` does not contain a `time` attribute.
            ValueError: if time stamps are not sorted.

        Returns:
            Slice of self.
        """
        # work in progress...
        raise NotImplementedError

        if "time" not in vars(self):
            msg = f"Object {self} has no time axis."
            raise TypeError(msg)

        if not np.all(np.sort(self.time) == self.time):
            msg = f"Time stamps for {self} are not sorted and therefore data cannot be selected by time."
            raise ValueError(msg)

        if start_time is None and end_time is None:
            warnings.warn("No starting or end timepoint was selected.")
            return self

        start_time = np.round(start_time, 7)
        end_time = np.round(end_time, 7)

        if start_time is None or start_time < self.time[0]:
            start_index = max(0, -n_before)

        # elif start_inclusive:
        #     start_index = bisect.bisect_right(self.time, start_time) - 1
        # else:
        #     start_index = bisect.bisect_left(self.time, start_time)

        # if end_time is None:
        #     end_index = len(self.time)
        # elif end_inclusive:
        #     end_index = bisect.bisect_left(self.time, end_time) + 1
        # else:
        #     end_index = bisect.bisect_left(self.time, end_time)

        # return self.select_by_index(
        #     start=start_index,
        #     end=end_index,
        #     label=label,
        # )


@dataclass
class TimeIndexer:
    """Helper class for slicing an object using the time axis instead of indices.

    Example:
    ```
    >>> sequence = load_eit_data(<path>, ...)
    >>> time_slice1 = sequence.t[tp_start:tp_end]
    >>> time_slice2 = sequence.select_by_time(tp_start, tp_end)
    >>> time_slice1 == time_slice2
    True
    ```
    """

    obj: SelectByTime

    def __getitem__(self, key: slice | float):
        if isinstance(key, slice):
            if key.step:
                msg = "Can't slice by time using specific step sizes."
                raise ValueError(msg)
            return self.obj.select_by_time(start_time=key.start, end_time=key.stop)

        if isinstance(key, int | float):
            return self.obj.select_by_time(start_time=key, end_time=key, end_inclusive=True)

        msg = f"Invalid slicing input. Should be `slice` or `int` or `float`, not {type(key)}."
        raise TypeError(msg)
