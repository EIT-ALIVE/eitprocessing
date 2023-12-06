from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from dataclasses import astuple
from dataclasses import is_dataclass
import numpy as np
from typing_extensions import Self


class Equivalence(ABC):
    # inspired by: https://stackoverflow.com/a/51743960/5170442
    def __eq__(self, other):
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

    @abstractmethod
    def isequivalent(
        self,
        other: Self,
        raise_: bool = False,
        checks: dict[str, bool] = None,
    ) -> bool:
        """Test whether the data structure between two objects are equivalent.

        Equivalence in this case means that objects are equal in all respects,
        except for data content. Generally data loaded from the same source with
        identical preprocessing will be equivalent.

        Args:
            other: object that will be compared to self.
            raise_:
                if False (default): return `False` if not equivalent;
                if `True`: raise `EquivalenceError` if not equivalence.
            checks: Dictionary of bools that will be checked. This dictionary can be
                defined in each child class individually.
                Defaults to None, meaning that only `type`s are compared.

        Raises:
            EquivalenceError: if `raise_ == True` and the objects are not equivalent.

        Returns:
            bool describing result of equivalence comparison.
        """
        # TODO: find out what correct way is to send extra argument to parent class without pissing off the linter
        if not checks:
            checks = {}

        if self == other:
            return True
        try:
            if type(self) is not type(other):
                raise EquivalenceError(
                    f"Classes don't match: {type(self)}, {type(other)}"
                )
            for msg, check in checks.items():
                if not check:
                    raise EquivalenceError(msg)
        except EquivalenceError as e:
            if raise_:
                raise e
            return False
        return True


class EquivalenceError(TypeError, ValueError):
    """Raised if objects are not equivalent.

    Equivalence in this case means that objects are equal in all respects,
    except for data content. Generally data loaded from the same source with
    identical preprocessing will be equivalent.
    """
