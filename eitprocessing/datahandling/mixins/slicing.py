from __future__ import annotations

import bisect
import copy
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


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
        """De facto implementation of the `__getitem__` function.

        This function can also be called directly to add a label to the sliced
        object. Otherwise a default label describing the slice and original
        object is attached.
        """
        if start is None and end is None:
            warnings.warn("No starting or end timepoint was selected.")
            return self

        start = start if start is not None else 0
        end = end if end is not None else len(self)
        newlabel = newlabel or self.label

        return self._sliced_copy(start_index=start, end_index=end, newlabel=newlabel)

    @abstractmethod
    def __len__(self) -> int: ...

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

    @abstractmethod
    def select_by_time(self, *args, **kwargs) -> Self: ...  # noqa: D102


class SelectByTime(SelectByIndex, HasTimeIndexer):
    """Adds methods for slicing by time rather than index."""

    def select_by_time(  # noqa: D417
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        start_inclusive: bool = False,
        end_inclusive: bool = False,
        label: str | None = None,
    ) -> Self:
        """Get a shortened copy of the object, starting from start_time and ending at end_time.

        Given a start and end time stamp (i.e. its value, not its index),
        return a slice of the original object, which must contain a time axis.

        Args:
            start_time: first time point to include. Defaults to first frame of sequence.
            end_time: last time point. Defaults to last frame of sequence.
            start_inclusive (default: `True`), end_inclusive (default `False`):
                these arguments control the behavior if the given time stamp
                does not match exactly with an existing time stamp of the input.
                if `True`: the given time stamp will be inside the sliced object.
                if `False`: the given time stamp will be outside the sliced object.
            label: Description. Defaults to None, which will create a label based
                on the original object label and the frames by which it is sliced.

        Raises:
            TypeError: if `self` does not contain a `time` attribute.
            ValueError: if time stamps are not sorted.

        Returns:
            A shortened copy of the object.
        """
        if not hasattr(self, "time"):
            msg = f"Object {self} has no time axis."
            raise TypeError(msg)

        if len(self.time) == 0:
            # TODO: make proper new instances when not slicing
            return copy.deepcopy(self)

        if start_time is None and end_time is None:
            warnings.warn("No starting or end timepoint was selected.")
            return self

        if not np.all(np.sort(self.time) == self.time):
            msg = f"Time stamps for {self} are not sorted and therefore data cannot be selected by time."
            raise ValueError(msg)

        if start_time is None or start_time < self.time[0]:
            start_index = 0
        elif start_inclusive:
            start_index = bisect.bisect_right(self.time, start_time) - 1
        else:
            start_index = bisect.bisect_left(self.time, start_time)

        if end_time is None:
            end_index = len(self.time)
        elif end_inclusive:
            end_index = bisect.bisect_left(self.time, end_time) + 1
        else:
            end_index = bisect.bisect_left(self.time, end_time)

        return self.select_by_index(
            start=start_index,
            end=end_index,
            newlabel=label,
        )


T = TypeVar("T", bound=HasTimeIndexer)


@dataclass
class TimeIndexer(Generic[T]):
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

    obj: T

    def __getitem__(self, key: slice | float) -> T:
        if isinstance(key, slice):
            if key.step:
                msg = "Can't slice by time using specific step sizes."
                raise ValueError(msg)
            return self.obj.select_by_time(start_time=key.start, end_time=key.stop)

        if isinstance(key, int | float):
            return self.obj.select_by_time(start_time=key, end_time=key, end_inclusive=True)

        msg = f"Invalid slicing input. Should be `slice` or `int` or `float`, not {type(key)}."
        raise TypeError(msg)
