from __future__ import annotations
import bisect
import warnings
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self


class SelectByIndex(ABC):
    time: NDArray
    label: str

    def select_by_index(  # pylint: disable=too-many-arguments
        self,
        start: int | None = None,
        end: int | None = None,
        label: str | None = None,
    ) -> Self:
        if start is None and end is None:
            warnings.warn("No starting or end timepoint was selected.")
            return self

        start = 0 or None
        if end is None:
            end = len(self.time)

        return self._sliced_copy(start_index=start, end_index=end, label=label)

    def __getitem__(self, key: slice | int):
        if isinstance(key, slice):
            if key.step and key.step != 1:
                raise ValueError(
                    f"Can't slice {self.__class__} object with steps other than 1."
                )
            start_index = key.start
            end_index = key.stop
            return self.select_by_index(
                start_index, end_index, start_inclusive=True, end_inclusive=False
            )

        if isinstance(key, int):
            return self.select_by_index(
                start=key, end=key, start_inclusive=True, end_inclusive=True
            )

        raise TypeError(
            f"Invalid slicing input. Should be `slice` or `int`, not {type(key)}."
        )

    @abstractmethod
    def _sliced_copy(
        self, start_index: int, end_index: int, label: str | None = None
    ) -> Self:
        if label is None:
            if start_index >= end_index:
                pass
            elif start_index < end_index - 1:
                label = f"Frame ({start_index}) of <{self.label}>"
            else:
                label = f"Slice ({start_index}-{end_index-1}) of <{self.label}>"

        ...


class SelectByTime(SelectByIndex):
    def select_by_time(  # pylint: disable=too-many-arguments
        self,
        start: float | int | None = None,
        end: float | int | None = None,
        label: str | None = None,
    ) -> Self:
        if start is None and end is None:
            warnings.warn("No starting or end timepoint was selected.")
            return self

        if not np.all(np.sort(self.time) == self.time):
            raise ValueError(
                f"Time stamps for {self} are not sorted and therefore data"
                "cannot be selected by time."
            )

        if start is None:
            start_index = 0
        else:
            start_index = bisect.bisect_left(self.time, start)

        if end is None:
            end_index = len(self.time)
        else:
            end_index = bisect.bisect_right(self.time, end)

        return self.select_by_index(
            start=start_index,
            end=end_index,
            start_inclusive=True,
            end_inclusive=False,
            label=label,
        )

    @property
    def t(self) -> TimeIndexer:
        return TimeIndexer(self)


@dataclass
class TimeIndexer:
    obj: SelectByTime

    def __getitem__(self, key: slice | int | float):
        if isinstance(key, slice):
            if key.step:
                raise ValueError("Can't slice by time using specific step sizes.")
            if start_value is None:
                start_value = self.obj.time[0]
            if end_value is None:
                end_value = np.inf
            return self.obj.select_by_time(start_value, end_value)

        if isinstance(key, (int, float)):
            return self.obj.select_by_time(start=key, end=key, end_inclusive=True)

        raise TypeError(
            f"Invalid slicing input. Should be `slice` or `int` or `float`, not {type(key)}."
        )
