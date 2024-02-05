from abc import ABC
from dataclasses import astuple, is_dataclass

import numpy as np
from typing_extensions import Self


class Equivalence(ABC):
    # inspired by: https://stackoverflow.com/a/51743960/5170442
    def __eq__(self, other: Self):
        if self is other:
            return True
        if is_dataclass(self):
            if self.__class__ is not other.__class__:
                return NotImplemented
            t1 = astuple(self)
            t2 = astuple(other)
            return all(Equivalence._array_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))
        return Equivalence._array_safe_eq(self, other)

    @staticmethod
    def _array_safe_eq(a, b) -> bool:
        """Check if a and b are equal, even if they are numpy arrays containing nans."""

        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        try:
            return object.__eq__(a, b)  # `a == b` could trigger an infinite loop
        except TypeError:
            return NotImplemented

