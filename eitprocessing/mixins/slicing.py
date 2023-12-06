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

        start = start or 0
        if end is None:
            end = len(self)

        if label is None:
            if start > end:
                label = f"No frames selected from <{self.label}>"
            elif start < end - 1:
                label = f"Frames ({start}-{end-1}) of <{self.label}>"
            else:
                label = f"Frame ({start}) of <{self.label}>"

        return self._sliced_copy(start_index=start, end_index=end, label=label)

    def __getitem__(self, key: slice | int):
        if isinstance(key, slice):
            if key.step and key.step != 1:
                raise ValueError(
                    f"Can't slice {self.__class__} object with steps other than 1."
                )
            start_index = key.start
            end_index = key.stop
            return self.select_by_index(start_index, end_index)

        if isinstance(key, int):
            return self.select_by_index(start=key, end=key + 1)

        raise TypeError(
            f"Invalid slicing input. Should be `slice` or `int`, not {type(key)}."
        )

    @abstractmethod
    def _sliced_copy(
        self,
        start_index: int,
        end_index: int,
        label: str,
    ) -> Self:
        ...


class SelectByTime(SelectByIndex):
    time: NDArray

    def select_by_time(  # pylint: disable=too-many-arguments
        self,
        start: float | int | None = None,
        end: float | int | None = None,
        start_inclusive: bool = True,
        end_inclusive: bool = False,
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

        if start is None or start < self.time[0]:
            start_index = 0
        elif start_inclusive:
            start_index = bisect.bisect_right(self.time, start) - 1
        else:
            start_index = bisect.bisect_left(self.time, start)

        if end is None:
            end_index = len(self.time)
        elif end_inclusive:
            end_index = bisect.bisect_left(self.time, end) + 1
        else:
            end_index = bisect.bisect_right(self.time, end)

        return self.select_by_index(
            start=start_index,
            end=end_index,
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
            return self.obj.select_by_time(key.start, key.stop)

        if isinstance(key, (int, float)):
            return self.obj.select_by_time(start=key, end=key, end_inclusive=True)

        raise TypeError(
            f"Invalid slicing input. Should be `slice` or `int` or `float`, not {type(key)}."
        )
