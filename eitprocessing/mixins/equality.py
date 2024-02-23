from __future__ import annotations

from abc import ABC
from dataclasses import astuple, is_dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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

    def isequivalent(self, other: Self, raise_: bool = False) -> bool:
        """Test whether the data structure between two objects are equivalent.

        Equivalence, in this case means that objects are compatible e.g. to be
        merged. Data content can vary, but e.g. the category of data (e.g.
        airway pressure, flow, tidal volume) and unit, etc., must match.


        Args:
            other: object that will be compared to self.
            raise_: sets this method's behavior in case of non-equivalence. If
                True, an `EquivalenceError` is raised, otherwise `False` is
                returned.

        Raises:
            EquivalenceError: if `raise_ == True` and the objects are not
            equivalent.

        Returns:
            bool describing result of equivalence comparison.
        """
        if self == other:
            return True

        try:
            # check whether types match
            if type(self) is not type(other):
                msg = f"Types don't match: {type(self)}, {type(other)}"
                raise EquivalenceError(msg)

            # check keys in collection
            if isinstance(self, dict):
                if set(self.keys()) != set(other.keys()):
                    msg = f"Keys don't match:\n\t{self.keys()},\n\t{other.keys()}"
                    raise EquivalenceError(msg)

                for key in self:
                    if not self[key].isequivalent(other[key], False):
                        msg = f"Data in {key} doesn't match: {self[key]}, {other[key]}"
                        raise EquivalenceError(msg)

            # check attributes of data
            else:
                self._check_equivalence: list[str]
                for attr in self._check_equivalence:
                    if (s := getattr(self, attr)) != (o := getattr(other, attr)):
                        raise f"{attr.capitalize()}s don't match: {s}, {o}"

        # raise or return if a check fails
        except EquivalenceError:
            if raise_:
                raise
            return False

        # if all checks pass
        return True


class EquivalenceError(TypeError, ValueError):
    """Raised if objects are not equivalent."""
