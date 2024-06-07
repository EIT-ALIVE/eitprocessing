from __future__ import annotations

from collections import UserDict
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self


class Equivalence:
    """Mixin class that adds an equality and equivalence check."""

    # inspired by: https://stackoverflow.com/a/51743960/5170442
    def __eq__(self, other: object) -> bool:
        if self is other:
            return True

        if type(self) is not type(other):
            return False

        if is_dataclass(self) and isinstance(self, Equivalence):
            return self._eq_dataclass(other)

        if isinstance(self, UserDict):
            return self._eq_userdict(other)

        return Equivalence._array_safe_eq(self, other)

    def _eq_dataclass(self, other: object) -> bool:
        """Compare two dataclasses for equality."""
        if not is_dataclass(self) or not is_dataclass(other):
            msg = "self or other is not a Dataclass"
            raise TypeError(msg)

        field_names = {field.name for field in fields(self)}
        if set(vars(self).keys()) != field_names or set(vars(other).keys()) != field_names:
            return False

        compare_fields = filter(lambda x: x.compare, fields(self))

        return all(
            Equivalence._array_safe_eq(getattr(self, field.name), getattr(other, field.name))
            for field in compare_fields
        )

    def _eq_userdict(self, other: object) -> bool:
        """Compare two userdicts for equality."""
        if not isinstance(self, UserDict) or not isinstance(other, UserDict):
            msg = "self or other is not a Userdict"
            raise TypeError(msg)

        if set(self.keys()) != set(other.keys()):
            return False
        return all(Equivalence.__eq__(self[key], other[key]) for key in set(self.keys()))

    @staticmethod
    def _array_safe_eq(a: Any, b: Any) -> bool:  # noqa: ANN401, PLR0911
        """Check if a and b are equal, even if they are numpy arrays containing nans."""
        if not isinstance(b, type(a)) and not isinstance(a, type(b)):
            return False

        if isinstance(a, np.ndarray):
            return np.shape(a) == np.shape(b) and np.array_equal(a, b, equal_nan=True)

        if not isinstance(a, Equivalence):
            return a == b

        if isinstance(
            a,
            dict,
        ):  # TODO: check whether this is still necessary for dicts #185
            return dict.__eq__(a, b)

        if isinstance(a, UserDict):
            return UserDict.__eq__(a, b)

        try:
            # `a == b` could trigger an infinite loop when called on an instance of Equivalence
            # object.__eq__() works for most objects, except those implemented seperately above
            return object.__eq__(a, b)
        except TypeError:
            return False

    def isequivalent(self, other: Self, raise_: bool = False) -> bool:  # noqa: C901, PLR0912
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
                if is_dataclass(self):
                    check_equivalence_fields = filter(
                        lambda x: "check_equivalence" in x.metadata and x.metadata["check_equivalence"],
                        fields(self),
                    )
                    attrs = [field.name for field in check_equivalence_fields]

                else:
                    self._check_equivalence: list[str]
                    attrs = self._check_equivalence

                for attr in attrs:
                    if (s := getattr(self, attr)) != (o := getattr(other, attr)):
                        msg = f"Attribute {attr} doesn't match: {s}, {o}"
                        raise EquivalenceError(msg)  # noqa: TRY301

        # raise or return if a check fails
        except EquivalenceError:
            if raise_:
                raise
            return False

        # if all checks pass
        return True


class EquivalenceError(TypeError, ValueError):
    """Raised if objects are not equivalent."""
