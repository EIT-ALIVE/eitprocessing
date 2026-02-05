# %%
"""Array-like interface over NamedTuple collections enabling NumPy slicing.

Motivation
----------
Some data is best represented with multiple related data points (e.g., the start, middle and end time of a breath). They
can be represented as NamedTuple instances; lightweight containers that group related fields together. Lists (or tuples)
of NamedTuples are more difficult to handle efficiently compared to NumPy arrays, especially in multi-dimensional cases.
When NamedTuples are collected inside NumPy arrays, however, they loose their NamedTuple context, removing the field
names and data types.

This module provides NamedTupleArray, a container that wraps homogeneous collections of NamedTuple instances into a
NumPy structured array, preserving the NamedTuple field names and types while enabling NumPy-style slicing and
field-wise access. It allows access to fields and even computed properties by name.

Key features
------------
- Homogeneous type checking: ensures all items share the same NamedTuple type.
- Safe field views: returns read-only views for direct field access.
- Property evaluation: computes per-item properties, resolving postponed
  annotations to pick appropriate NumPy dtypes.
- Shape preservation: supports nested sequences, maintaining their shape in the
  structured array.
- Interop: from_ndarray helper to map last-axis columns to NamedTuple fields.

Example:
```python
class Coordinate(NamedTuple):
    x: float
    y: float
    z: float

    @property
    def r(self) -> float:
        \"\"\"The radial distance from the origin.\"\"\"
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

coords = [Coordinate(1.0, 2.0, 2.0), Coordinate(3.0, 4.0, 0.0), Coordinate(0.0, 0.0, 5.0)]
arr = NamedTupleArray(coords)

arr[1:]              # Slice of Coordinates
# NamedTupleArray[Coordinate](array([(3., 4., 0.), (0., 0., 5.)],
#       dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')]))
arr[0]               # Access a single Coordinate
# Coordinate(x=1.0, y=2.0, z=2.0)
arr["x"]             # Access x field across all Coordinates
# array([1., 3., 0.])
arr["r"]             # Access computed property across all Coordinates
# array([3., 5., 5.])
```
"""

from __future__ import annotations

import contextlib
from typing import (
    TYPE_CHECKING,
    Generic,
    NamedTuple,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import numpy as np

from eitprocessing.utils.frozen_array import freeze_array

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Sequence as SequenceType

    from numpy._core.multiarray import flagsobj


T = TypeVar("T", bound=tuple)
NonStringSeq: TypeAlias = tuple[T, ...] | list[T]
Nested = T | NonStringSeq


class NamedTupleArray(Generic[T]):
    """An array-like container for homogeneous NamedTuple instances.

    Overview
    --------
    NamedTupleArray wraps a sequence (or nested sequence) of NamedTuple items
    into a NumPy structured ndarray, enabling:
    - NumPy-style indexing and slicing (preserving shape).
    - Field access by name that returns read-only NumPy views.
    - Computation of per-item properties or attributes, returning a NumPy array
      with dtype inferred from the property's type annotation when available.
    - Immutability control: pass frozen=True to prevent all array modifications.

    Construction
    ------------
    - From a sequence: NamedTupleArray([nt1, nt2, ...])
      Validates that all items are of the same NamedTuple type.
      Nested sequences are supported; their shape is preserved.
    - From an ndarray: NamedTupleArray.from_ndarray(arr, NTType)
      'arr' must have last axis equal to the number of fields in NTType.
      Columns along the last axis are mapped to NamedTuple fields.
    - Use NamedTupleArray(..., frozen=True) or NamedTupleArray.from_ndarray(..., frozen=True) to make the array
      immutable (prevents all modifications).

    Access
    ------
    - Field by name: arr["x"] -> returns a read-only view of the field.
    - Property/attribute: arr["duration"] -> computes the property for each
      item, producing an ndarray. If the property has a type hint, the result
      dtype is chosen accordingly (e.g., int -> int64, float -> float64).
      Otherwise, a heuristic casts ints to int64, else tries float, else object.
    - Direct array access via items property: arr.items (always returns the
      underlying NumPy array; if frozen=True, modifications are prevented).

    Notes:
    -----
    - Homogeneity: All elements must be the same NamedTuple type.
    - String fields are kept as object dtype to avoid truncation.
    - Properties are evaluated per element; heavy properties may be costly.
    - Field views are always read-only to prevent accidental mutation.
    - The .items property returns the underlying array; modifications are only
      prevented if frozen=True was passed during construction.

    Example:
    -------
    >>> class Breath(NamedTuple):
    ...     start_time: float
    ...     middle_time: float
    ...     end_time: float
    ...     @property
    ...     def duration(self) -> float:
    ...         return self.end_time - self.start_time
    ...
    >>> breaths = [Breath(0.0, 0.5, 1.0), Breath(1.0, 1.6, 2.1)]
    >>> arr = NamedTupleArray(breaths)
    >>> arr["duration"]
    array([1. , 1.1])
    """

    namedtuple_type: type[T]
    __items: np.ndarray

    def __init__(
        self,
        items: NonStringSeq[T] | np.ndarray | Nested[T],
        namedtuple_type: type[T] | None = None,
        frozen: bool = True,
    ):
        """Initialize a NamedTupleArray from a sequence or nested sequence of NamedTuple items.

        Args:
            items: A sequence (or nested sequence) of NamedTuple instances, or a numpy ndarray containing them.
            namedtuple_type:
                Optional explicit type of the NamedTuple items. If not provided, it will be inferred from the first leaf
                item.
            frozen: If True (default), makes the underlying array immutable.
        """
        if namedtuple_type is not None:
            self.namedtuple_type = namedtuple_type
        else:
            if isinstance(items, np.ndarray) and items.size == 0:
                msg = "Cannot infer type from empty array."
                raise ValueError(msg)
            if not isinstance(items, np.ndarray) and not items:
                msg = "Cannot infer type from empty sequence."
                raise ValueError(msg)

            # Infer NT type from first leaf element
            leaf = _first_leaf(items)
            self.namedtuple_type = type(leaf)  # type: ignore[assignment]

        # Validate homogeneity
        _check_homogeneous(items, self.namedtuple_type)

        # Build structured dtype and array with same shape
        dt = _get_tuple_dtype(self.namedtuple_type)
        self.__items = np.asarray(items, dtype=dt)

        if frozen:
            self._freeze()

        object.__setattr__(self, "_initialized", True)

    def _freeze(self) -> None:
        """Make the underlying array immutable."""
        dt = _get_tuple_dtype(self.namedtuple_type)
        freeze_method = "flag" if dt.hasobject else "memoryview"
        self.__items = freeze_array(self.__items, method=freeze_method)

    def __setattr__(self, name: str, value: object) -> None:
        """Allow setting type and __items only during initialization; block modification after."""
        # Check if initialization is complete (use object.__getattribute__ to bypass our __getattr__)
        try:
            initialized = object.__getattribute__(self, "_initialized")
        except AttributeError:
            initialized = False

        # Allow setting type and __items only during initialization
        if not initialized and name in ("namedtuple_type", "_NamedTupleArray__items"):
            super().__setattr__(name, value)
        elif initialized and name in ("namedtuple_type", "_NamedTupleArray__items"):
            msg = f"{type(self).__name__!r} object is immutable; cannot modify {name!r} after initialization."
            raise AttributeError(msg)
        else:
            msg = f"{type(self).__name__!r} object is immutable; cannot set attribute {name!r}."
            raise AttributeError(msg)

    @classmethod
    def from_array(cls, arr: np.ndarray | Nested, namedtuple_type: type[T], frozen: bool = True) -> NamedTupleArray[T]:
        """Build a NamedTupleArray from an unstructured numpy array or nested list.

        The list must be convertible to a numpy array. The last axis of the array is mapped to the fields.
        The length of the last axis must equal to the number of fields in the given NamedTuple type.

        Example:
            This examples represents a sequence of 10 breaths for each of 32x32 pixels. Each breath contains 3 fields:
            start_time, middle_time, end_time.

            ```python
            breath_data = load_breath_data()  # shape (10, 32, 32, 3)
            breaths = NamedTupleArray.from_ndarray(breath_data, Breath)
            ```

            This is equivalent to a list of 10 nested lists, each containing 32 lists (rows) of 32 (columns) Breath
            objects.
        """
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        if arr.ndim < 1:
            msg = "arr must have at least 1 dimension."
            raise ValueError(msg)
        n_fields = len(namedtuple_type._fields)
        if (lal := arr.shape[-1]) != n_fields:
            msg = f"Last axis must have size {n_fields} for {namedtuple_type.__name__}, not {lal}."
            raise ValueError(msg)
        dt = _get_tuple_dtype(namedtuple_type)
        out = np.empty(arr.shape[:-1], dtype=dt)

        if not dt.fields:
            msg = "Generated dtype has no fields; cannot proceed."
            raise RuntimeError(msg)

        fields = cast("dict[str, tuple[np.dtype, int]]", dt.fields)
        for i, name in enumerate(namedtuple_type._fields):
            # Cast each column to the target field dtype to avoid unintended promotion
            target_dt = fields[name][0]
            out[name] = arr[..., i].astype(target_dt, copy=False)

        inst = cls.__new__(cls)
        inst.namedtuple_type = namedtuple_type
        inst._NamedTupleArray__items = out  # noqa: SLF001

        if frozen:
            inst._freeze()  # noqa: SLF001

        return inst

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the NamedTupleArray."""
        return self.__items.shape

    @property
    def ndim(self) -> int:
        """The number of dimensions of the NamedTupleArray."""
        return self.__items.ndim

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the underlying structured array."""
        return self.__items.dtype

    @property
    def items(self) -> np.ndarray:
        """The underlying NumPy structured array.

        Returns the private array. If this instance was created with frozen=True,
        modifications via this reference are prevented. Otherwise, modifications
        are allowed.
        """
        return self.__items

    @property
    def flags(self) -> flagsobj:
        """The flags of the underlying structured array.

        If this instance was created with frozen=True, the WRITEABLE flag cannot
        be changed. Otherwise, the flags are fully mutable.
        """
        return self.__items.flags

    def to_array(self) -> np.ndarray:
        """Convert to an unstructured numpy array.

        Returns a 2D array where each column corresponds to a field of the NamedTuple,
        in the order of the NamedTuple fields. This allows convenient slicing by
        column indices like `arr[:, [0, 2]]`.

        Returns:
            A 2D unstructured numpy array of shape (n_items, n_fields).

        Example:
            >>> class Point(NamedTuple):
            ...     x: float
            ...     y: float
            ...     z: float
            >>> nta = NamedTupleArray([Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)])
            >>> arr = nta.to_array()
            >>> arr.shape
            (2, 3)
            >>> arr[:, [0, 2]]  # Get x and z columns
            array([[1., 3.],
                   [4., 6.]])
        """
        # Stack each field as a column to create unstructured array
        if not self.__items.dtype.names:
            # No fields, return empty array
            return np.empty((self.shape[0], 0))

        return np.column_stack([self.__items[name] for name in self.__items.dtype.names])

    def __getattr__(self, attr: str):
        """Block access to the private array.

        All array attributes should be accessed via explicit properties.
        This prevents users from bypassing immutability controls.
        """
        msg = f"{type(self).__name__!r} object has no attribute {attr!r}"
        raise AttributeError(msg)

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        return self.__items.astype(dtype) if dtype is not None else self.__items

    def __iter__(self) -> Generator[T | NamedTupleArray[T], None, None]:
        if self.ndim == 1:
            for item in self.__items:
                yield self.namedtuple_type(*item)  # type: ignore[call-arg]
        else:
            # yield structured subarrays along axis 0
            for i in range(self.__items.shape[0]):
                out = NamedTupleArray.__new__(NamedTupleArray)
                out.namedtuple_type = self.namedtuple_type
                out._NamedTupleArray__items = self.__items[i]  # noqa: SLF001
                yield out

    def __len__(self) -> int:
        return self.__items.shape[0] if self.__items.ndim > 0 else 0

    def __repr__(self) -> str:
        return f"NamedTupleArray[{self.namedtuple_type.__name__}]{repr(self.__items).removeprefix('array')}"

    def __eq__(self, other: object) -> bool:
        """Compare two NamedTupleArray instances for equality.

        Two NamedTupleArray instances are equal if:
        - They are both NamedTupleArray instances
        - They have the same NamedTuple type
        - Their underlying arrays are equal (including NaN equality for floats)
        """
        if not isinstance(other, NamedTupleArray):
            return False

        if self.namedtuple_type is not other.namedtuple_type:
            return False

        # Compare shapes
        if self.__items.shape != other.__items.shape:
            return False

        # Compare dtypes
        if self.__items.dtype != other.__items.dtype:
            return False

        # For structured arrays, compare field by field to handle NaN values properly
        for name in self.__items.dtype.names or []:
            self_field = self.__items[name]
            other_field = other.__items[name]

            # Use array_equal with equal_nan for each field
            if not np.array_equal(self_field, other_field, equal_nan=True):
                return False

        return True

    __hash__ = None  # type: ignore[assignment]

    @overload
    def __getitem__(self, index: str) -> np.ndarray: ...

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> NamedTupleArray[T]: ...

    @overload
    def __getitem__(self, index: NonStringSeq) -> NamedTupleArray[T]: ...

    def __getitem__(self, index: str | int | slice | NonStringSeq) -> np.ndarray | NamedTupleArray[T] | T:
        # Field-name access: return field view
        if isinstance(index, str):
            names = self.__items.dtype.names or ()
            if index in names:
                view = self.__items[index]
                # Ensure field view is read-only
                with contextlib.suppress(Exception):
                    view.flags.writeable = False
                return view
            # Computed property or attribute on the NT → compute over all items
            return self._compute_property(index)

        # NumPy-style indexing
        result = self.__items[index]

        # Structured scalar (np.void) → return NamedTuple
        if isinstance(result, np.void):
            # For structured np.void, convert to NamedTuple
            return self.namedtuple_type(*result.tolist())  # type: ignore[call-arg]

        # Zero-d structured ndarray (shape == ()) → convert to NamedTuple
        if isinstance(result, np.ndarray) and result.dtype.fields is not None and result.ndim == 0:
            scalar = result.item()  # np.void
            return self.namedtuple_type(*scalar.tolist())  # type: ignore[call-arg]

        # Structured ndarray → wrap
        if isinstance(result, np.ndarray) and result.dtype.fields is not None:
            out: NamedTupleArray[T] = type(self).__new__(type(self))
            out.namedtuple_type = self.namedtuple_type
            out._NamedTupleArray__items = result
            return out

        # Non-structured ndarray (e.g. field slice) → return as-is
        return result

    def _compute_property(self, attr: str) -> np.ndarray:
        """Compute a property or attribute across all items, preserving the array shape."""
        # Verify attribute exists on the NT instance
        sample = self.namedtuple_type(*self.__items.flat[0].tolist())  # type: ignore[call-arg]
        if not hasattr(sample, attr):
            msg = f"Field or property '{attr}' not found in NamedTuple."
            raise KeyError(msg)

        # Collect values (single pass using flat indexing)
        out_obj = np.empty(self.shape, dtype=object)
        for i, rec in enumerate(self.__items.reshape(-1)):
            nt = self.namedtuple_type(*rec.tolist())  # type: ignore[call-arg]
            out_obj.reshape(-1)[i] = getattr(nt, attr)

        # Determine target dtype from property annotation if available (handles postponed annotations)
        target_dtype: np.dtype | None = None
        attr_member = getattr(self.namedtuple_type, attr, None)
        if isinstance(attr_member, property) and attr_member.fget is not None:
            with contextlib.suppress(Exception):
                hints = get_type_hints(attr_member.fget)
                ret_ann = hints.get("return")
                if ret_ann is not None:
                    target = _python_to_np_dtype(ret_ann)
                    target_dtype = np.dtype(target)

        # Cast accordingly
        if target_dtype is not None and target_dtype != np.dtype(object):
            with contextlib.suppress(Exception):
                return out_obj.astype(target_dtype)

        # Heuristics: ints → int64; floats → float64; numpy scalar families respected
        with contextlib.suppress(Exception):
            if all(isinstance(v, (int, np.integer)) for v in out_obj.flat):
                return out_obj.astype(np.int64)
        with contextlib.suppress(Exception):
            if all(isinstance(v, (float, np.floating)) for v in out_obj.flat):
                return out_obj.astype(np.float64)

        return out_obj


def _first_leaf(
    seq: NamedTuple | np.ndarray | list[NamedTuple] | tuple[NamedTuple, ...] | Nested[NamedTuple],
) -> NamedTuple:
    """Recursively find the first NamedTuple instance in a nested sequence or ndarray."""
    if _is_namedtuple_instance(seq):
        return seq
    if isinstance(seq, np.ndarray):
        if seq.size == 0:
            msg = "Cannot infer type from empty ndarray."
            raise ValueError(msg)
        return _first_leaf(seq.flat[0])
    if isinstance(seq, (list, tuple)):
        if not seq:
            msg = "Cannot infer type from empty nested sequence."
            raise ValueError(msg)
        return _first_leaf(seq[0])

    msg = "Items must be NamedTuple or nested sequences thereof."
    raise TypeError(msg)


def _check_homogeneous(seq: SequenceType[NamedTuple] | np.ndarray | Nested[T], typ: type[T]) -> None:
    """Recursively check that all NamedTuple instances in the nested sequence/ndarray are of the given type."""
    if isinstance(seq, np.ndarray):
        for it in seq.flat:
            _check_homogeneous(it, typ)
        return
    if isinstance(seq, (list, tuple)) and not _is_namedtuple_instance(seq):
        seq_ = cast("SequenceType[NamedTuple | Nested[T]]", seq)
        for it in seq_:
            it: NamedTuple | Nested[T]
            _check_homogeneous(it, typ)
        return
    if _is_namedtuple_instance(seq):
        if type(seq) is not typ:
            msg = "All items must be of the same NamedTuple type."
            raise ValueError(msg)
        return
    msg = "Items must be NamedTuple or nested sequences thereof."
    raise TypeError(msg)


def _python_to_np_dtype(py: type) -> np.dtype | str:
    """Map basic Python types to NumPy dtypes."""
    if py is int:
        return "i8"
    if py is float:
        return "f8"
    if py is bool:
        return "?"
    if py is str:
        return np.dtype(object)  # keep Python str; avoids truncation

    if get_origin(py) is Union:
        args = [a for a in get_args(py) if a is not type(None)]
        if len(args) == 1:
            return _python_to_np_dtype(args[0])
    return np.dtype(object)


def _is_namedtuple_instance(item: object) -> TypeGuard[NamedTuple]:
    """Check if item is a NamedTuple instance."""
    return isinstance(item, tuple) and hasattr(item, "_fields")


def _is_namedtuple_type(item: object) -> TypeGuard[type[NamedTuple]]:
    """Check if item is a NamedTuple type."""
    return isinstance(item, type) and issubclass(item, tuple) and hasattr(item, "_fields")


def _get_tuple_dtype(item: NamedTuple | type[tuple]) -> np.dtype:
    """Generate a NumPy structured dtype from a NamedTuple type."""
    if _is_namedtuple_instance(item):
        item = type(item)
    if not _is_namedtuple_type(item):
        msg = "item must be a NamedTuple or a NamedTuple type."
        raise TypeError(msg)

    hints = get_type_hints(item, include_extras=False)
    names_in_order = list(item._fields)

    def to_np_dtype(py: type) -> str | np.dtype:
        if py is int:
            return "i8"
        if py is float:
            return "f8"
        if py is bool:
            return "?"
        # everything else (incl. str) → object to avoid truncation
        return np.dtype(object)

    fields = [(name, to_np_dtype(hints.get(name, object))) for name in names_in_order]
    return np.dtype(fields)
