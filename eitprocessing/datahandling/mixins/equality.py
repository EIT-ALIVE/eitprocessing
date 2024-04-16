from __future__ import annotations

from collections import UserDict
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


class Equivalence:
    """Mixin class that adds an equality and equivalence check."""

    # inspired by: https://stackoverflow.com/a/51743960/5170442
    def __eq__(self, other: Self) -> bool:
        if self is other:
            return True

        if type(self) is not type(other):
            return False

        if is_dataclass(self):
            attrs_self = Equivalence._remove_naming_info(vars(self))
            attrs_other = Equivalence._remove_naming_info(vars(other))
            if set(attrs_self.keys()) != set(attrs_other.keys()):
                return False
            return all(Equivalence._array_safe_eq((attrs_self[k], attrs_other[k]) for k in attrs_self))
        return Equivalence._array_safe_eq(self, other)

    @staticmethod
    def _array_safe_eq(a: Any, b: Any) -> bool:  # noqa: ANN401, PLR0911
        """Check if a and b are equal, even if they are numpy arrays containing nans."""
        if not isinstance(b, type(a)):
            return NotImplemented

        if isinstance(a, np.ndarray):
            return np.shape(a) == np.shape(b) and np.array_equal(a, b, equal_nan=True)

        if not isinstance(a, Equivalence):
            return a == b

        if isinstance(a, dict):  # TODO: check whether this is still necessary for dicts #185
            return dict.__eq__(a, b)

        if isinstance(a, UserDict):
            return UserDict.__eq__(a, b)

        try:
            # `a == b` could trigger an infinite loop when called on an instance of Equivalence
            # object.__eq__() works for most objects, except those implemented seperately above
            return object.__eq__(a, b)
        except TypeError:
            return False

    @staticmethod
    def _remove_naming_info(d: dict | UserDict) -> dict:
        x = ["label", "name", "description"]
        return {k: v for k, v in d.items() if k not in x}

    def isequivalent(self, other: Self, raise_: bool = False) -> bool:  # noqa: C901
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
                raise EquivalenceError(msg)  # noqa: TRY301

            # check keys in collection
            # TODO: check whether this is still necessary for dicts #185
            if isinstance(self, dict | UserDict):
                if set(self.keys()) != set(other.keys()):
                    msg = f"Keys don't match:\n\t{self.keys()},\n\t{other.keys()}"
                    raise EquivalenceError(msg)  # noqa: TRY301

                for key in self:
                    if not self[key].isequivalent(other[key], False):
                        msg = f"Data in {key} doesn't match: {self[key]}, {other[key]}"
                        raise EquivalenceError(msg)  # noqa: TRY301

            # check attributes of data
            else:
                self._check_equivalence: list[str]
                for attr in self._check_equivalence:
                    if (s := getattr(self, attr)) != (o := getattr(other, attr)):
                        raise f"Attribute {attr} doesn't match: {s}, {o}"

        # raise or return if a check fails
        except EquivalenceError:
            if raise_:
                raise
            return False

        # if all checks pass
        return True


class EquivalenceError(TypeError, ValueError):
    """Raised if objects are not equivalent."""
