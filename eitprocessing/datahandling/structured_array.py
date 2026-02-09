# %%
"""Array-like interface over NamedTuple and dataclass collections enabling NumPy slicing.

Motivation
----------
Some data is best represented with multiple related data points (e.g., the start, middle and end time of a breath). They
can be represented as NamedTuple or dataclass instances; lightweight containers that group related fields together.
Lists (or tuples) of such instances are more difficult to handle efficiently compared to NumPy arrays, especially in
multi-dimensional cases. When such instances are collected inside NumPy arrays, however, they lose their context,
removing the field names and data types.

This module provides StructuredArray, a container that wraps homogeneous collections of NamedTuple or dataclass
instances into a NumPy structured array, preserving field names and types while enabling NumPy-style slicing and
field-wise access. It allows access to fields and even computed properties by name. Dataclass instances benefit from
automatic __post_init__ validation on access.

Key features
------------
- Homogeneous type checking: ensures all items share the same NamedTuple or dataclass type.
- Safe field views: returns read-only views for direct field access.
- Property evaluation: computes per-item properties, resolving postponed
  annotations to pick appropriate NumPy dtypes.
- Shape preservation: supports nested sequences, maintaining their shape in the
  structured array.
- Validation support: dataclass __post_init__ is called on item access.
- Interop: from_array helper to map array columns to fields.

Example with NamedTuple:
```python
from typing import NamedTuple

class Coordinate(NamedTuple):
    x: float
    y: float
    z: float

    @property
    def r(self) -> float:
        \"\"\"The radial distance from the origin.\"\"\"
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

coords = [Coordinate(1.0, 2.0, 2.0), Coordinate(3.0, 4.0, 0.0), Coordinate(0.0, 0.0, 5.0)]
arr = StructuredArray(coords)

arr[0]               # Access a single Coordinate
# Coordinate(x=1.0, y=2.0, z=2.0)
arr["x"]             # Access x field across all Coordinates
# array([1., 3., 0.])
arr["r"]             # Access computed property across all Coordinates
# array([3., 5., 5.])
```

Example with dataclass:
```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

    def __post_init__(self):
        if self.x < 0 or self.y < 0:
            raise ValueError("Coordinates must be non-negative")

points = [Point(1.0, 2.0), Point(3.0, 4.0)]
arr = StructuredArray(points)  # Validation runs on item construction/access
arr[0]  # Point(x=1.0, y=2.0)
```
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import fields as dc_fields
from dataclasses import is_dataclass
from pathlib import Path
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


T = TypeVar("T", bound=tuple | object)  # NamedTuple or dataclass instance
NonStringSeq: TypeAlias = tuple[T, ...] | list[T]
Nested = T | NonStringSeq

# Mutable types that should trigger warnings when used in frozen dataclasses
MUTABLE_TYPES: tuple[type, ...] = (list, dict, set, bytearray, np.ndarray)

# Immutable types that are allowed in dataclass fields
# Note: np.generic is included to allow all numpy scalar types (np.int64, np.float32, etc.)
ALLOWED_IMMUTABLE_TYPES: tuple[type, ...] = (str, int, float, bool, bytes, tuple, frozenset, Path, np.generic)

# All allowed types (used for disallowed type checking)
ALLOWED_TYPES: tuple[type, ...] = ALLOWED_IMMUTABLE_TYPES + MUTABLE_TYPES


class StructuredArray(Generic[T]):
    """An array-like container for homogeneous NamedTuple or dataclass instances.

    Overview
    --------
    StructuredArray wraps a sequence (or nested sequence) of NamedTuple or dataclass items
    into a NumPy structured ndarray, enabling:
    - NumPy-style indexing and slicing (preserving shape).
    - Field access by name that returns read-only NumPy views.
    - Computation of per-item properties or attributes, returning a NumPy array
      with dtype inferred from the property's type annotation when available.
    - Immutability control: pass frozen=True to prevent all array modifications.
    - Validation support: dataclass __post_init__ runs on item access.

    Construction
    ------------
    - From a sequence: StructuredArray([item1, item2, ...])
      Validates that all items are of the same NamedTuple or dataclass type.
      Type is inferred from the first item. For empty sequences, provide item_type explicitly.
    - From a sequence with explicit type: StructuredArray([...], item_type=MyType)
      Allows type specification upfront, enabling empty sequences and ensuring type safety.
    - From an ndarray: StructuredArray.from_array(arr, ItemType)
      'arr' must have last axis equal to the number of fields in ItemType.
      Columns along the last axis are mapped to item fields.
    - Use StructuredArray(..., frozen=True) or StructuredArray.from_array(..., frozen=True) to make the array
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
    - Homogeneity: All elements must be the same NamedTuple or dataclass type.
    - String fields are kept as object dtype to avoid truncation.
    - Properties are evaluated per element; heavy properties may be costly.
    - Field views are always read-only to prevent accidental mutation.
    - The .items property returns the underlying array; modifications are only
      prevented if frozen=True was passed during construction.
    - For dataclasses: __post_init__ is called when accessing items via iteration
      or indexing to ensure validation happens.

    Example with NamedTuple:
    -------
    >>> from typing import NamedTuple
    >>> class Breath(NamedTuple):
    ...     start_time: float
    ...     middle_time: float
    ...     end_time: float
    ...     @property
    ...     def duration(self) -> float:
    ...         return self.end_time - self.start_time
    ...
    >>> breaths = [Breath(0.0, 0.5, 1.0), Breath(1.0, 1.6, 2.1)]
    >>> arr = StructuredArray(breaths)
    >>> arr["duration"]
    array([1. , 1.1])

    Example with dataclass:
    -------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Event:
    ...     time: float
    ...     value: float
    ...
    ...     def __post_init__(self):
    ...         if self.value < 0:
    ...             raise ValueError("value must be positive")
    ...
    >>> events = [Event(0.0, 1.0), Event(1.0, 2.0)]
    >>> arr = StructuredArray(events)
    >>> arr[0]  # Runs __post_init__ validation
    Event(time=0.0, value=1.0)

    Example with explicit type for empty list:
    -------
    >>> # Create an empty StructuredArray with explicit type
    >>> arr = StructuredArray([], item_type=Event)
    >>> len(arr)
    0
    """

    item_type: type[T]
    _items: np.ndarray
    _is_dataclass: bool
    frozen: bool

    def __init__(
        self,
        items: NonStringSeq[T] | np.ndarray | Nested[T],
        item_type: type[T] | None = None,
        frozen: bool = True,
    ):
        """Initialize a StructuredArray from a sequence or nested sequence of items.

        Args:
            items: A sequence (or nested sequence) of NamedTuple or dataclass instances,
                   or a numpy ndarray containing them.
            item_type:
                Optional explicit type of the items. If provided, all items are validated
                against this type. If not provided, the type is inferred from the first
                leaf item. Required when items is empty.
            frozen: If True (default), makes the underlying array immutable.

        Raises:
            ValueError: If items is empty and item_type is None, or if homogeneity check fails.
            TypeError: If items contain unsupported types.
        """
        if item_type is not None:
            self.item_type = item_type
        else:
            if isinstance(items, np.ndarray) and items.size == 0:
                msg = "Cannot infer type from empty array. Provide item_type explicitly."
                raise ValueError(msg)
            if not isinstance(items, np.ndarray) and not items:
                msg = "Cannot infer type from empty sequence. Provide item_type explicitly."
                raise ValueError(msg)

            # Infer type from first leaf element
            leaf = _first_leaf(items)
            self.item_type = type(leaf)  # type: ignore[assignment]

        # Validate homogeneity
        _check_homogeneous(items, self.item_type)

        # Determine if item_type is a dataclass
        self._is_dataclass = is_dataclass(self.item_type)

        # Build structured dtype and array with same shape
        dt = _get_struct_dtype(self.item_type)

        # For dataclasses, numpy.asarray doesn't know how to convert them directly,
        # so we convert to tuples first (via named tuples or tuples of field values)
        if self._is_dataclass:
            items = cast("NonStringSeq[T] | np.ndarray | Nested[T]", _dataclass_items_to_tuples(items, self.item_type))

        self._items = np.asarray(items, dtype=dt)

        # Check for disallowed field types in dataclass
        if self._is_dataclass:
            disallowed_fields = _get_disallowed_field_names(self.item_type)
            if disallowed_fields:
                warnings.warn(
                    f"Dataclass '{self.item_type.__name__}' has fields with disallowed types: "
                    f"{', '.join(disallowed_fields)}. "
                    f"Allowed types are: str, int, float, bool, bytes, tuple, frozenset, pathlib.Path, "
                    f"numpy types, frozen dataclasses, and NamedTuples.",
                    UserWarning,
                    stacklevel=2,
                )

        self.frozen = frozen
        if frozen:
            # Check if dataclass is not frozen while array is
            if self._is_dataclass and not _is_dataclass_frozen(self.item_type):
                warnings.warn(
                    f"StructuredArray is frozen, but the underlying dataclass "
                    f"'{self.item_type.__name__}' is not. Items accessed from the array "
                    f"can still be modified. Consider freezing the dataclass with "
                    f"@dataclass(frozen=True).",
                    UserWarning,
                    stacklevel=2,
                )
            # Check if dataclass has mutable fields (independent check - can warn even if dataclass is frozen)
            if self._is_dataclass and _has_mutable_fields(self.item_type):
                warnings.warn(
                    f"StructuredArray is frozen, but the dataclass '{self.item_type.__name__}' has mutable fields "
                    f"(list, dict, set, etc.). The contents of these fields can still be modified. "
                    f"Consider using immutable types (tuple) instead.",
                    UserWarning,
                    stacklevel=2,
                )
            self._freeze()

        object.__setattr__(self, "_initialized", True)

    def _freeze(self) -> None:
        """Make the underlying array immutable."""
        dt = _get_struct_dtype(self.item_type)
        freeze_method = "flag" if dt.hasobject else "memoryview"
        self._items = freeze_array(self._items, method=freeze_method)

    def __setattr__(self, name: str, value: object) -> None:
        """Allow setting type and _items only during initialization; block modification after."""
        # Check if initialization is complete (use object.__getattribute__ to bypass our __getattr__)
        try:
            initialized = object.__getattribute__(self, "_initialized")
        except AttributeError:
            initialized = False

        # Allow setting type, _items, and _is_dataclass only during initialization
        if not initialized and name in ("item_type", "_items", "_is_dataclass", "frozen"):
            super().__setattr__(name, value)
        elif initialized and name in ("item_type", "_items", "_is_dataclass", "frozen"):
            msg = f"{type(self).__name__!r} object is immutable; cannot modify {name!r} after initialization."
            raise AttributeError(msg)
        else:
            msg = f"{type(self).__name__!r} object is immutable; cannot set attribute {name!r}."
            raise AttributeError(msg)

    @classmethod
    def from_array(cls, arr: np.ndarray | Nested, item_type: type[T], frozen: bool = True) -> StructuredArray[T]:  # noqa: C901
        """Build a StructuredArray from an unstructured numpy array or nested list.

        The list must be convertible to a numpy array. The last axis of the array is mapped to the fields.
        The length of the last axis must equal to the number of fields in the given type.

        Example:
            This examples represents a sequence of 10 breaths for each of 32x32 pixels. Each breath contains 3 fields:
            start_time, middle_time, end_time.

            ```python
            breath_data = load_breath_data()  # shape (10, 32, 32, 3)
            breaths = StructuredArray.from_array(breath_data, Breath)
            ```

            This is equivalent to a list of 10 nested lists, each containing 32 lists (rows) of 32 (columns) items.
        """
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        if arr.ndim < 1:
            msg = "array must have at least 1 dimension."
            raise ValueError(msg)

        n_fields = _get_n_fields(item_type)
        if (lal := arr.shape[-1]) != n_fields:
            msg = f"Last axis must have size {n_fields} for {item_type.__name__}, not {lal}."
            raise ValueError(msg)

        dt = _get_struct_dtype(item_type)
        out = np.empty(arr.shape[:-1], dtype=dt)

        if not dt.fields:
            msg = "Generated dtype has no fields; cannot proceed."
            raise RuntimeError(msg)

        fields = cast("dict[str, tuple[np.dtype, int]]", dt.fields)
        field_names = _get_field_names(item_type)
        for i, name in enumerate(field_names):
            # Cast each column to the target field dtype to avoid unintended promotion
            target_dt = fields[name][0]
            out[name] = arr[..., i].astype(target_dt, copy=False)

        inst = cls.__new__(cls)
        inst.item_type = item_type
        inst._is_dataclass = is_dataclass(item_type)  # noqa: SLF001
        inst._items = out  # noqa: SLF001

        if inst._is_dataclass and "__post_init__" in dir(inst.item_type):  # noqa: SLF001
            # Run __post_init__ for all items to ensure validation
            for record in inst._items.flat:  # noqa: SLF001
                inst._reconstruct_item(record)  # noqa: SLF001

        # Check for disallowed field types in dataclass
        if inst._is_dataclass:  # noqa: SLF001
            disallowed_fields = _get_disallowed_field_names(item_type)
            if disallowed_fields:
                warnings.warn(
                    f"Dataclass '{item_type.__name__}' has fields with disallowed types: "
                    f"{', '.join(disallowed_fields)}. "
                    "Allowed types are: str, int, float, bool, bytes, tuple, frozenset, pathlib.Path, "
                    "numpy types, frozen dataclasses, and NamedTuples.",
                    UserWarning,
                    stacklevel=2,
                )

        inst.frozen = frozen
        if frozen:
            # Check if dataclass is not frozen while array is
            if inst._is_dataclass and not _is_dataclass_frozen(item_type):  # noqa: SLF001
                warnings.warn(
                    "StructuredArray is frozen, but the underlying dataclass "
                    f"'{item_type.__name__}' is not. Consider freezing the dataclass with "
                    "@dataclass(frozen=True).",
                    UserWarning,
                    stacklevel=2,
                )
            # Check if dataclass has mutable fields (independent check - can warn even if dataclass is frozen)
            if inst._is_dataclass and _has_mutable_fields(item_type):  # noqa: SLF001
                warnings.warn(
                    f"StructuredArray is frozen, but the dataclass '{item_type.__name__}' has mutable fields "
                    f"(list, dict, set, etc.). The contents of these fields can still be modified. "
                    f"Consider using immutable types (tuple) instead.",
                    UserWarning,
                    stacklevel=2,
                )
            inst._freeze()  # noqa: SLF001

        object.__setattr__(inst, "_initialized", True)

        return inst

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the StructuredArray."""
        return self._items.shape

    @property
    def ndim(self) -> int:
        """The number of dimensions of the StructuredArray."""
        return self._items.ndim

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the underlying structured array."""
        return self._items.dtype

    @property
    def items(self) -> np.ndarray:
        """The underlying NumPy structured array.

        Returns the private array. If this instance was created with frozen=True,
        modifications via this reference are prevented. Otherwise, modifications
        are allowed.
        """
        return self._items

    @property
    def flags(self) -> flagsobj:
        """The flags of the underlying structured array.

        If this instance was created with frozen=True, the WRITEABLE flag cannot
        be changed. Otherwise, the flags are fully mutable.
        """
        return self._items.flags

    def to_array(self) -> np.ndarray:
        """Convert to an unstructured numpy array.

        Returns a 2D array where each column corresponds to a field of the item type,
        in field order. This allows convenient slicing by column indices like
        `arr[:, [0, 2]]`.

        Returns:
            A 2D unstructured numpy array of shape (n_items, n_fields).

        Example:
            >>> from typing import NamedTuple
            >>> class Point(NamedTuple):
            ...     x: float
            ...     y: float
            ...     z: float
            >>> arr = StructuredArray([Point(1.0, 2.0, 3.0), Point(4.0, 5.0, 6.0)])
            >>> arr_2d = arr.to_array()
            >>> arr_2d.shape
            (2, 3)
            >>> arr_2d[:, [0, 2]]  # Get x and z columns
            array([[1., 3.],
                   [4., 6.]])
        """
        # Stack each field as a column to create unstructured array
        if not self._items.dtype.names:
            # No fields, return empty array
            return np.empty((self.shape[0], 0))

        return np.column_stack([self._items[name] for name in self._items.dtype.names])

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        return self._items.astype(dtype) if dtype is not None else self._items

    def __iter__(self) -> Generator[T | StructuredArray[T], None, None]:
        if self.ndim == 1:
            for item in self._items:
                yield self._reconstruct_item(item)
        else:
            # yield structured subarrays along axis 0
            for i in range(self._items.shape[0]):
                out = StructuredArray.__new__(StructuredArray)
                out.item_type = self.item_type
                out._is_dataclass = self._is_dataclass  # noqa: SLF001
                out._items = self._items[i]  # noqa: SLF001
                yield out

    def __len__(self) -> int:
        return self._items.shape[0] if self._items.ndim > 0 else 0

    def __repr__(self) -> str:
        return f"StructuredArray[{self.item_type.__name__}]{repr(self._items).removeprefix('array')}"

    def __eq__(self, other: object) -> bool:
        """Compare two StructuredArray instances for equality.

        Two StructuredArray instances are equal if:
        - They are both StructuredArray instances
        - They have the same item type
        - Their underlying arrays are equal (including NaN equality for floats)
        """
        if not isinstance(other, StructuredArray):
            return False

        if self.item_type is not other.item_type:
            return False

        # Compare shapes
        if self._items.shape != other._items.shape:
            return False

        # Compare dtypes
        if self._items.dtype != other._items.dtype:
            return False

        # For structured arrays, compare field by field to handle NaN values properly
        for name in self._items.dtype.names or []:
            self_field = self._items[name]
            other_field = other._items[name]

            # Use array_equal with equal_nan for each field
            if not np.array_equal(self_field, other_field, equal_nan=True):
                return False

        return True

    __hash__ = None  # type: ignore[assignment]

    def __add__(self, other: StructuredArray[T]) -> StructuredArray[T]:
        if not isinstance(other, StructuredArray):
            msg = f"Can only concatenate StructuredArray (not '{type(other).__name__}') to StructuredArray."
            raise TypeError(msg)

        if self.item_type is not other.item_type:
            msg = "Cannot concatenate StructuredArray with different item types."
            raise TypeError(msg)

        new_items = np.concatenate((self._items, other._items), axis=0)
        frozen = self.frozen or other.frozen

        # Create a new StructuredArray directly with the concatenated structured array
        inst = self.__class__.__new__(self.__class__)
        inst.item_type = self.item_type
        inst._is_dataclass = self._is_dataclass
        inst._items = new_items
        inst.frozen = frozen

        if frozen:
            inst._freeze()

        object.__setattr__(inst, "_initialized", True)
        return inst

    @overload
    def __getitem__(self, index: str) -> np.ndarray: ...

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> StructuredArray[T]: ...

    @overload
    def __getitem__(self, index: NonStringSeq) -> StructuredArray[T]: ...

    def __getitem__(self, index: str | int | slice | NonStringSeq) -> np.ndarray | StructuredArray[T] | T:
        # Field-name access: return field view
        if isinstance(index, str):
            names = self._items.dtype.names or ()
            if index in names:
                view = self._items[index]
                # Ensure field view is read-only
                with contextlib.suppress(Exception):
                    view.flags.writeable = False
                return view
            # Computed property or attribute on the item type → compute over all items
            return self._compute_property(index)

        # NumPy-style indexing
        result = self._items[index]

        # Structured scalar (np.void) → return reconstructed item
        if isinstance(result, np.void):
            return self._reconstruct_item(result)

        # Zero-d structured ndarray (shape == ()) → convert to item
        if isinstance(result, np.ndarray) and result.dtype.fields is not None and result.ndim == 0:
            scalar = result.item()  # np.void
            return self._reconstruct_item(scalar)

        # Structured ndarray → wrap
        if isinstance(result, np.ndarray) and result.dtype.fields is not None:
            out: StructuredArray[T] = type(self).__new__(type(self))
            out.item_type = self.item_type
            out._is_dataclass = self._is_dataclass
            out._items = result
            out.frozen = self.frozen
            object.__setattr__(out, "_initialized", True)
            return out

        # Non-structured ndarray (e.g. field slice) → return as-is
        return result

    def _reconstruct_item(self, record: np.void) -> T:
        """Reconstruct an item (NamedTuple or dataclass) from a numpy void record.

        For dataclasses, this calls __post_init__ to ensure validation runs.
        """
        values = record.tolist()

        return self.item_type(*values)

    def _compute_property(self, attr: str) -> np.ndarray:
        """Compute a property or attribute across all items, preserving the array shape."""
        # Verify attribute exists on a sample item
        sample_record = self._items.flat[0]
        sample = self._reconstruct_item(sample_record)
        if not hasattr(sample, attr):
            msg = f"Field or property '{attr}' not found in {self.item_type.__name__}."
            raise KeyError(msg)

        # Collect values (single pass using flat indexing)
        out_obj = np.empty(self.shape, dtype=object)
        for i, rec in enumerate(self._items.reshape(-1)):
            item = self._reconstruct_item(rec)
            out_obj.reshape(-1)[i] = getattr(item, attr)

        # Determine target dtype from property annotation if available (handles postponed annotations)
        target_dtype: np.dtype | None = None
        attr_member = getattr(self.item_type, attr, None)
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
) -> object:
    """Recursively find the first NamedTuple or dataclass instance in a nested sequence or ndarray."""
    if _is_struct_instance(seq):
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

    msg = "Items must be NamedTuple or dataclass or nested sequences thereof."
    raise TypeError(msg)


def _check_homogeneous(seq: SequenceType | np.ndarray | Nested[T], typ: type[T]) -> None:
    """Recursively check that all items in the nested sequence/ndarray are of the given type."""
    if isinstance(seq, np.ndarray):
        for it in seq.flat:
            _check_homogeneous(it, typ)
        return
    if isinstance(seq, (list, tuple)) and not _is_struct_instance(seq):
        seq_ = cast("SequenceType", seq)
        for it in seq_:
            _check_homogeneous(it, typ)
        return
    if _is_struct_instance(seq):
        if type(seq) is not typ:
            msg = f"All items must be of the same type ({typ.__name__}), got {type(seq).__name__}."
            raise ValueError(msg)
        return
    msg = "Items must be NamedTuple, dataclass, or nested sequences thereof."
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
    return isinstance(item, tuple) and hasattr(item, "_fields") and not is_dataclass(item)


def _dataclass_items_to_tuples(items: object, dc_type: type) -> object:
    """Recursively convert dataclass items to tuples for numpy.asarray conversion.

    numpy.asarray doesn't know how to convert arbitrary dataclass instances
    to structured array rows, so we convert them to tuples first.
    """
    if isinstance(items, np.ndarray):
        # For ndarrays, convert each element
        result = np.array([_dataclass_items_to_tuples(item, dc_type) for item in items.flat])
        return result.reshape(items.shape)

    if isinstance(items, (list, tuple)) and not is_dataclass(items):
        # It's a sequence - recursively convert each element
        return [_dataclass_items_to_tuples(item, dc_type) for item in items]

    # It's a single dataclass instance - convert to tuple of field values
    if is_dataclass(items):
        field_names = _get_field_names(type(items))
        return tuple(getattr(items, name) for name in field_names)

    # Shouldn't reach here, but just in case
    return items


def _is_namedtuple_type(item: object) -> TypeGuard[type[NamedTuple]]:
    """Check if item is a NamedTuple type."""
    return isinstance(item, type) and issubclass(item, tuple) and hasattr(item, "_fields") and not is_dataclass(item)


def _is_struct_instance(item: object) -> bool:
    """Check if item is a NamedTuple or dataclass instance (not type)."""
    if isinstance(item, type):
        # Exclude types themselves
        return False
    return _is_namedtuple_instance(item) or is_dataclass(item)


def _is_struct_type(item: object) -> bool:
    """Check if item is a NamedTuple or dataclass type."""
    return _is_namedtuple_type(item) or (isinstance(item, type) and is_dataclass(item))


def _get_field_names(item_type: type) -> list[str]:
    """Get field names from a NamedTuple or dataclass type."""
    if is_dataclass(item_type):
        return [f.name for f in dc_fields(item_type)]
    if _is_namedtuple_type(item_type):
        return list(item_type._fields)  # type: ignore[return-value]
    msg = f"item_type must be a NamedTuple or dataclass type, got {item_type}"
    raise TypeError(msg)


def _get_n_fields(item_type: type) -> int:
    """Get number of fields from a NamedTuple or dataclass type."""
    return len(_get_field_names(item_type))


def _get_struct_dtype(item: object) -> np.dtype:
    """Generate a NumPy structured dtype from a NamedTuple or dataclass type."""
    if _is_struct_instance(item):
        item = type(item)
    if not _is_struct_type(item):
        msg = "item must be a NamedTuple instance, NamedTuple type, dataclass instance, or dataclass type."
        raise TypeError(msg)

    if is_dataclass(item):
        return _get_dataclass_dtype(item)  # type: ignore[arg-type]
    return _get_namedtuple_dtype(item)  # type: ignore[arg-type]


def _get_namedtuple_dtype(nt_type: type[NamedTuple]) -> np.dtype:
    """Generate a NumPy structured dtype from a NamedTuple type."""
    hints = get_type_hints(nt_type, include_extras=False)
    names_in_order = list(nt_type._fields)  # type: ignore[attr-defined]

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


def _get_dataclass_dtype(dc_type: type) -> np.dtype:
    """Generate a NumPy structured dtype from a dataclass type."""
    try:
        hints = get_type_hints(dc_type, include_extras=False)
    except (NameError, TypeError, AttributeError):
        # If we can't resolve type hints (e.g., for types defined in test scopes),
        # fall back to empty hints
        hints = {}

    def to_np_dtype(py: type) -> str | np.dtype:
        if py is int:
            return "i8"
        if py is float:
            return "f8"
        if py is bool:
            return "?"
        # everything else (incl. str) → object to avoid truncation
        return np.dtype(object)

    fields = [(f.name, to_np_dtype(hints.get(f.name, object))) for f in dc_fields(dc_type)]
    return np.dtype(fields)


def _is_dataclass_frozen(dc_type: type) -> bool:
    """Check if a dataclass type is frozen.

    Args:
        dc_type: A dataclass type.

    Returns:
        True if the dataclass is frozen, False otherwise.
    """
    if not is_dataclass(dc_type):
        return False

    # Access the __dataclass_params__ which contains the frozen flag
    if hasattr(dc_type, "__dataclass_params__"):
        return dc_type.__dataclass_params__.frozen  # type: ignore[attr-defined]

    return False


def _has_mutable_fields(dc_type: type) -> bool:
    """Check if a dataclass has any mutable field types.

    A frozen dataclass with mutable fields like list, dict, or set can still have
    its contents modified even though the dataclass instance itself is frozen.

    Args:
        dc_type: A dataclass type.

    Returns:
        True if the dataclass has any mutable field types, False otherwise.
    """
    if not is_dataclass(dc_type):
        return False

    try:
        hints = get_type_hints(dc_type, include_extras=False)
    except (NameError, TypeError, AttributeError):
        # If we can't resolve type hints, skip the check
        return False

    mutable_types = MUTABLE_TYPES

    for field_type in hints.values():
        # Get the origin type (e.g., list from list[int])
        origin = get_origin(field_type)

        # Check if directly mutable or is a subclass of mutable types
        if _is_subclass_of_allowed(field_type, mutable_types) or (
            origin is not None and _is_subclass_of_allowed(origin, mutable_types)
        ):
            return True

        # Check if it's a Union with mutable types
        if origin is Union:
            args = get_args(field_type)
            for arg in args:
                arg_origin = get_origin(arg)
                if _is_subclass_of_allowed(arg, mutable_types) or (
                    arg_origin is not None and _is_subclass_of_allowed(arg_origin, mutable_types)
                ):
                    return True

    return False


def _get_disallowed_field_names(dc_type: type) -> list[str]:
    """Get field names that have disallowed types in a dataclass.

    Allowed types include: str, int, float, bool, bytes, tuple, frozenset, pathlib.Path,
    numpy scalars/arrays, frozen dataclasses, and NamedTuples.

    Args:
        dc_type: A dataclass type.

    Returns:
        List of field names with disallowed types, empty if all types are allowed.
    """
    if not is_dataclass(dc_type):
        return []

    try:
        hints = get_type_hints(dc_type, include_extras=False)
    except (NameError, TypeError, AttributeError):
        # If we can't resolve type hints, skip the check
        return []

    disallowed_fields = []

    # All allowed types (immutable and mutable)
    allowed = ALLOWED_TYPES

    for field_name, field_type in hints.items():
        origin = get_origin(field_type)

        # Check if directly allowed or is a subclass of allowed types
        if _is_subclass_of_allowed(field_type, allowed):
            continue

        # Check if it's a generic type with allowed origin
        if origin is not None and _is_subclass_of_allowed(origin, allowed):
            continue

        # Check for Union types
        if origin is Union:
            args = get_args(field_type)
            if all(_is_allowed_type(arg) for arg in args if arg is not type(None)):
                continue

        # Check for frozen dataclass or NamedTuple
        if _is_namedtuple_type(field_type) or (is_dataclass(field_type) and _is_dataclass_frozen(field_type)):
            continue

        # If we get here, the type is not allowed
        disallowed_fields.append(field_name)

    return disallowed_fields


def _is_subclass_of_allowed(check_type: type, allowed_types: tuple[type, ...]) -> bool:
    """Check if check_type is a subclass of any type in allowed_types."""
    if check_type in allowed_types:
        return True

    try:
        if isinstance(check_type, type):
            for allowed in allowed_types:
                if allowed is not type and issubclass(check_type, allowed):
                    return True
    except TypeError:
        # issubclass() can raise TypeError for non-class types
        pass

    return False


def _is_allowed_type(field_type: type) -> bool:
    """Check if a field type is in the allowed list or is a subclass of an allowed type."""
    # Check against basic allowed types (includes numpy types via np.generic)
    if _is_subclass_of_allowed(field_type, ALLOWED_TYPES):
        return True

    if _is_namedtuple_type(field_type):
        return True

    if is_dataclass(field_type) and _is_dataclass_frozen(field_type):
        return True

    # Check for generic types
    origin = get_origin(field_type)
    return origin is not None and _is_subclass_of_allowed(origin, ALLOWED_TYPES)
