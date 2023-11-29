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
    def select_by_index(  # pylint: disable=too-many-arguments
        self,
        start: int | None = None,
        end: int | None = None,
        start_inclusive: bool = True,
        end_inclusive: bool = False,
        label: str | None = None,
    ) -> Self:
        if not any((start, end)):
            warnings.warn("No starting or end timepoint was selected.")
            return self

        if start is None:
            start = 0

        if end is None:
            end = len(self.time)

        if not start_inclusive:
            start += 1

        if end_inclusive:
            end += 1

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
            return self.select_by_index(start=key, end=key, end_inclusive=True)

        raise TypeError(f"Invalid key type. Should be slice or int, not {type(key)}.")

    @abstractmethod
    def _sliced_copy(
        self, start_index: int, end_index: int, label: str | None = None
    ) -> Self:
        ...


class SelectByTime(SelectByIndex):
    time: NDArray

    def _check_time_sorted(self):
        if not np.all(np.sort(self.time) == self.time):
            raise ValueError(
                f"Time stamps for {self} are not sorted and therefor data"
                "cannot be selected by time."
            )

    def select_by_time(  # pylint: disable=too-many-arguments
        self,
        start: float | int | None = None,
        end: float | int | None = None,
        start_inclusive: bool = True,
        end_inclusive: bool = False,
        label: str | None = None,
    ) -> Self:
        if not any((start, end)):
            warnings.warn("No starting or end timepoint was selected.")
            return self

        self._check_time_sorted()

        if start is None:
            start_index = 0
        elif start_inclusive:
            start_index = bisect.bisect_left(self.time, start)
        else:
            start_index = bisect.bisect_right(self.time, start)

        if end is None:
            end_index = None
        elif end_inclusive:
            end_index = bisect.bisect_right(self.time, end) - 1
        else:
            end_index = bisect.bisect_left(self.time, end) - 1

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
            start_value = key.start or self.obj.time[0]
            end_value = key.stop or np.inf
            return self.obj.select_by_time(start_value, end_value)
        if isinstance(key, (int, float)):
            return self.obj.select_by_time(start=key, end=key, end_inclusive=True)
        raise TypeError(
            f"Invalid key type. Should be slice, int or float, not {type(key)}."
        )
