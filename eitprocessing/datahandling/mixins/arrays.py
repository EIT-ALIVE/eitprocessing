from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from typing_extensions import Never


class NotAnArray:
    """Mixin class that prevents numpy and scipy from converting an object to an array.

    Objects in this package keep their numeric data in an attribute (e.g. `values`), rather than being arrays
    themselves. Passing such an object directly to a numpy or scipy function is virtually always a mistake, but numpy
    does not treat it as one: it falls back to the sequence protocol (`__len__`/`__getitem__`) or wraps the object in a
    0-dimensional object array. For sliceable objects that returns a copy of the object for every index, which is
    prohibitively slow; for other objects it silently produces an object array that does not contain the data.

    This mixin closes the three routes numpy uses, so any attempt raises a `TypeError` explaining what to pass instead:

    - `__array__` is used by `numpy.asarray()`/`numpy.array()`, and therefore by most of scipy, which converts its
      input before doing anything else;
    - `__array_ufunc__` is used by ufuncs, e.g. `numpy.sin()`, `numpy.add()` and `array + object`;
    - `__array_function__` is used by the rest of the numpy API, e.g. `numpy.mean()` and `numpy.concatenate()`, which
      dispatches before any conversion happens.

    Regular Python behaviour (slicing, `len()`, iteration, comparison) is unaffected.

    The error message points at the field holding the data, if there is one. Mark that field with
    `field(metadata={"array_attribute": True})` to have it named.
    """

    @property
    def _array_attribute(self) -> str | None:
        """Name of the field holding the data, or None if no field is marked as such.

        Only looked up when an error is raised, so the cost of scanning the fields does not matter.
        """
        if not is_dataclass(self):
            return None
        return next((field.name for field in fields(self) if field.metadata.get("array_attribute")), None)

    def _refuse_array_conversion(self, attempted: str = "") -> Never:
        """Raise a `TypeError` explaining that this object can not be used as an array."""
        msg = f"`{type(self).__name__}` objects can not be used as an array{attempted}."
        if self._array_attribute:
            msg += f" Pass the `{self._array_attribute}` attribute instead."
        raise TypeError(msg)

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> Never:
        self._refuse_array_conversion()

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Never:  # noqa: ANN401
        self._refuse_array_conversion(f" (attempted `numpy.{ufunc.__name__}`)")

    def __array_function__(self, func: Callable, types: Any, args: Any, kwargs: Any) -> Never:  # noqa: ANN401
        self._refuse_array_conversion(f" (attempted `{func.__module__}.{func.__name__}`)")
