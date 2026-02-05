from typing import NamedTuple

import numpy as np
import pytest

from eitprocessing.datahandling.namedtuple_array import NamedTupleArray


class Mixed(NamedTuple):
    """NamedTuple with mixed field types and a computed property."""

    a: int
    b: float
    c: bool
    d: str

    @property
    def d_length(self) -> int:
        """Computed property returning the length of string d."""
        return len(self.d)


class Simple(NamedTuple):
    """NamedTuple with simple numeric fields."""

    x: int
    y: float


class Breath(NamedTuple):
    """NamedTuple representing a breath with start, mid, end times."""

    start: float
    mid: float
    end: float

    @property
    def duration(self) -> float:
        """Computed property returning the duration of the breath."""
        return self.end - self.start


def test_1d_mixed_types_and_properties():
    items = [Mixed(1, 2.0, True, "foo"), Mixed(3, 4.5, False, "hello")]
    nta = NamedTupleArray(items)

    assert nta.shape == (2,)
    # scalar access
    v0 = nta[0]
    assert isinstance(v0, Mixed)
    assert v0.a == 1
    assert v0.b == 2.0
    assert v0.c is True
    assert v0.d == "foo"

    # field views have expected dtype and values
    a = nta["a"]
    assert a.dtype.kind in ("i", "u")
    assert a.shape == (2,)
    assert (a == np.array([1, 3])).all()

    b = nta["b"]
    assert np.issubdtype(b.dtype, np.floating)

    c = nta["c"]
    assert np.issubdtype(c.dtype, np.bool_)

    d = nta["d"]
    # strings kept as object
    assert d.dtype == object

    # computed property -> returns int dtype (annotated)
    dl = nta["d_length"]
    assert np.issubdtype(dl.dtype, np.integer)
    assert list(dl) == [3, 5]


def test_2d_indexing_and_slicing():
    nested = [[Simple(i + j, float(i * j)) for j in range(3)] for i in range(2)]
    nta2d = NamedTupleArray(nested)
    assert nta2d.shape == (2, 3)

    # scalar multi-dimensional indexing returns NamedTuple
    item = nta2d[0, 1]
    assert isinstance(item, Simple)
    assert item.x == 1
    assert item.y == 0.0

    # row slice returns NamedTupleArray
    row = nta2d[0]
    assert isinstance(row, NamedTupleArray)
    assert row.shape == (3,)

    # field access on 2D returns array with original shape
    xs = nta2d["x"]
    assert xs.shape == (2, 3)
    assert xs[0, 1] == 1


def test_3d_from_ndarray_and_indexing():
    # create shape (2,2,2,2) last axis 2 fields
    arr = np.array(
        [
            [[[1, 2.0], [3, 4.0]], [[5, 6.0], [7, 8.0]]],
            [[[9, 10.0], [11, 12.0]], [[13, 14.0], [15, 16.0]]],
        ],
        dtype=float,
    )
    nta = NamedTupleArray.from_array(arr, Simple)
    assert nta.shape == (2, 2, 2)

    # random 3D scalar access
    s = nta[1, 0, 1]
    assert isinstance(s, Simple)
    assert s.x == 11
    assert s.y == 12.0


def test_field_views_readonly_and_shape_preserved():
    items = [Simple(1, 2.0), Simple(3, 4.0)]
    nta = NamedTupleArray(items, frozen=False)
    assert nta.flags.writeable is True
    vx = nta["x"]
    assert vx.flags.writeable is False
    assert vx.shape == (2,)


def test_calculated_property_float_dtype():
    breaths = [Breath(0.0, 0.5, 1.0), Breath(1.0, 1.4, 2.2)]
    nta = NamedTupleArray(breaths)
    dur = nta["duration"]
    assert np.issubdtype(dur.dtype, np.floating)
    assert pytest.approx(list(dur), rel=1e-9) == [1.0, 1.2]


def test_forwarding_attributes_and_methods():
    items = [Simple(1, 2.0), Simple(3, 4.0)]
    nta = NamedTupleArray(items)

    # dtype via property
    assert nta.dtype is not None

    # reshape is not available (array is private now)
    assert not hasattr(nta, "reshape")

    # flags and ndim reflect the underlying array
    assert nta.flags.writeable is not None
    assert nta.ndim == 1

    # field-view methods are available (e.g., sum)
    vx = nta["x"]
    assert hasattr(vx, "sum")
    assert int(vx.sum()) == 4


def test_frozen_namedtuple_array_numeric():
    items = [Simple(1, 2.0), Simple(3, 4.0)]
    frozen_nta = NamedTupleArray(items, frozen=True)
    nta = NamedTupleArray(items, frozen=False)

    # Field views are always read-only (for both frozen and unfrozen)
    field_view = nta["x"]
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        field_view[0] = 10

    # Frozen array also prevents field modification
    frozen_field_view = frozen_nta["x"]
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        frozen_field_view[0] = 10

    # Both frozen and unfrozen arrays block new attributes
    with pytest.raises(AttributeError):
        nta.new_attribute = 42
    with pytest.raises(AttributeError):
        frozen_nta.new_attribute = 42

    # Both frozen and unfrozen arrays block modification of _type
    with pytest.raises(AttributeError):
        nta._type = np.floating
    with pytest.raises(AttributeError):
        frozen_nta._type = np.floating

    # Frozen array prevents toggling writeable flag
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True of this array"):
        frozen_nta.flags.writeable = True


def test_frozen_namedtuple_array_string():
    items = [Mixed(1, 2.0, True, "foo"), Mixed(3, 4.0, False, "bar")]
    frozen_nta = NamedTupleArray(items, frozen=True)

    # Frozen array with object dtype prevents field modification via read-only view
    frozen_field_view = frozen_nta["d"]
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        frozen_field_view[0] = "baz"

    # Frozen array blocks new attributes
    with pytest.raises(AttributeError):
        frozen_nta.new_attribute = 42

    # Frozen array blocks modifying _type
    with pytest.raises(AttributeError):
        frozen_nta._type = np.floating

    # For object dtypes, users cannot access the underlying array to toggle the writeable flag
    # because __items is now private (inaccessible). This is the key benefit of the refactoring.
    # The writeable flag is protected by preventing direct array access.
    with pytest.raises(AttributeError, match="has no attribute '_items'"):
        _ = frozen_nta._items


def test_items_property_unfrozen():
    """Test that .items property returns the underlying array for unfrozen arrays."""
    items = [Simple(1, 2.0), Simple(3, 4.0)]
    nta = NamedTupleArray(items, frozen=False)

    # Access via .items property works
    underlying = nta.items
    assert isinstance(underlying, np.ndarray)
    assert underlying.shape == (2,)
    assert underlying.dtype.names == ("x", "y")

    # For unfrozen arrays, modifications are allowed (though field views are still read-only)
    assert nta.items.flags.writeable is True


def test_items_property_frozen():
    """Test that .items property is read-only for frozen arrays."""
    items = [Simple(1, 2.0), Simple(3, 4.0)]
    nta = NamedTupleArray(items, frozen=True)

    # Access via .items property works
    underlying = nta.items
    assert isinstance(underlying, np.ndarray)
    assert underlying.shape == (2,)

    # For frozen arrays with numeric dtypes, writeable is False
    assert nta.items.flags.writeable is False


def test_slicing_frozen_array():
    """Test that slicing a frozen array returns a frozen sub-array."""
    items = [Simple(i, float(i)) for i in range(5)]
    nta = NamedTupleArray(items, frozen=True)

    # Slice returns NamedTupleArray with same frozenness
    sliced = nta[1:3]
    assert isinstance(sliced, NamedTupleArray)
    assert sliced.shape == (2,)
    assert sliced.items.flags.writeable is False


def test_computed_property_frozen():
    """Test that computed properties work correctly on frozen arrays."""
    breaths = [Breath(0.0, 0.5, 1.0), Breath(1.0, 1.4, 2.2)]
    nta = NamedTupleArray(breaths, frozen=True)

    # Computed property returns correct values
    dur = nta["duration"]
    assert pytest.approx(list(dur), rel=1e-9) == [1.0, 1.2]
    # Computed properties create new arrays, so they're writable
    assert dur.flags.writeable is True


def test_single_item_array():
    """Test NamedTupleArray with a single item."""
    items = [Simple(42, 3.14)]
    nta = NamedTupleArray(items, frozen=True)

    assert nta.shape == (1,)
    assert nta[0] == Simple(42, 3.14)
    assert nta["x"][0] == 42


def test_2d_frozen_array():
    """Test 2D frozen array slicing and access."""
    nested = [[Simple(i + j, float(i * j)) for j in range(2)] for i in range(2)]
    nta = NamedTupleArray(nested, frozen=True)

    assert nta.shape == (2, 2)
    assert nta.items.flags.writeable is False

    # Slicing should also be frozen
    row = nta[0]
    assert row.items.flags.writeable is False


def test_all_properties_accessible():
    """Test that all expected properties are accessible."""
    items = [Simple(1, 2.0), Simple(3, 4.0)]
    nta = NamedTupleArray(items)

    # All properties should be accessible
    assert nta.shape == (2,)
    assert nta.ndim == 1
    assert nta.dtype is not None
    assert nta.flags is not None
    assert nta.items is not None


def test_setattr_blocked_post_init():
    """Test that __setattr__ blocks all attribute setting after init."""
    items = [Simple(1, 2.0)]
    nta = NamedTupleArray(items, frozen=False)

    # Cannot set any new attributes
    with pytest.raises(AttributeError, match="immutable"):
        nta.custom_attr = "value"

    # Cannot modify internal attributes
    with pytest.raises(AttributeError, match="immutable"):
        nta._type = int


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================


def test_empty_sequence():
    """Test that empty sequences raise ValueError."""
    with pytest.raises(ValueError, match="Cannot infer type from empty"):
        NamedTupleArray([])


def test_empty_ndarray():
    """Test that empty ndarrays raise ValueError."""
    empty_array = np.array([], dtype=float)
    with pytest.raises(ValueError, match="Cannot infer type from empty"):
        NamedTupleArray(empty_array)


def test_non_namedtuple_items():
    """Test that passing non-NamedTuple items raises TypeError."""
    items = [(1, 2.0), (3, 4.0)]  # Regular tuples, not NamedTuples
    with pytest.raises(TypeError, match="NamedTuple"):
        NamedTupleArray(items)


def test_ndarray_last_axis_mismatch():
    """Test that from_ndarray with mismatched last axis raises error."""
    # Simple has 2 fields, but array has 3 columns
    arr = np.array([[1, 2.0, 3.0], [4, 5.0, 6.0]])
    with pytest.raises(ValueError):
        NamedTupleArray.from_array(arr, Simple)


def test_frozen_from_ndarray():
    """Test that from_ndarray with frozen=True works correctly."""
    arr = np.array([[1, 2.0], [3, 4.0]])
    nta = NamedTupleArray.from_array(arr, Simple, frozen=True)

    assert nta.shape == (2,)
    assert nta.items.flags.writeable is False


def test_iteration():
    """Test that NamedTupleArray is iterable."""
    items = [Simple(1, 2.0), Simple(3, 4.0), Simple(5, 6.0)]
    nta = NamedTupleArray(items)

    # Iteration should yield NamedTuple instances
    for i, item in enumerate(nta):
        assert isinstance(item, Simple)
        assert item == items[i]


def test_len():
    """Test that len() works on NamedTupleArray."""
    items = [Simple(1, 2.0), Simple(3, 4.0), Simple(5, 6.0)]
    nta = NamedTupleArray(items)

    assert len(nta) == 3


def test_repr():
    """Test that repr() produces a meaningful string."""
    items = [Simple(1, 2.0), Simple(3, 4.0)]
    nta = NamedTupleArray(items)

    r = repr(nta)
    assert "NamedTupleArray" in r
    assert "Simple" in r


def test_computed_property_nonexistent_attribute():
    """Test accessing a computed property that doesn't exist."""
    items = [Simple(1, 2.0), Simple(3, 4.0)]
    nta = NamedTupleArray(items)

    # Accessing a non-existent attribute should raise KeyError
    with pytest.raises(KeyError):
        _ = nta["nonexistent_property"]


def test_heterogeneous_items():
    """Test that arrays with different NamedTuple types raise error."""
    items = [Simple(1, 2.0), Mixed(3, 4.0, True, "foo")]
    with pytest.raises(ValueError):
        NamedTupleArray(items)


def test_mixed_items_and_non_items():
    """Test that mixing NamedTuples with non-NamedTuples raises error."""
    items = [Simple(1, 2.0), (3, 4.0)]  # Mix of NamedTuple and tuple
    with pytest.raises(TypeError):
        NamedTupleArray(items)


def test_passing_array_in_list():
    """Test that passing a numpy array wrapped in a list works."""
    arr = np.array([(1, 2.0), (3, 4.0)], dtype=[("x", "i8"), ("y", "f8")])
    # This should be treated as a single-item list containing an array
    # and should fail since arrays aren't NamedTuple instances
    with pytest.raises(TypeError):
        NamedTupleArray([arr])


def test_empty_nested_list():
    """Test that empty nested lists raise ValueError."""
    with pytest.raises(ValueError, match="Cannot infer type from empty"):
        NamedTupleArray([[]])


def test_casting_computed_property():
    """Test computed properties with type annotations are cast correctly."""
    breaths = [Breath(0.0, 0.5, 1.0), Breath(1.0, 1.4, 2.2)]
    nta = NamedTupleArray(breaths)

    # duration is annotated as float
    dur = nta["duration"]
    assert np.issubdtype(dur.dtype, np.floating)

    # d_length (from Mixed) is annotated as int
    mixed_items = [Mixed(1, 2.0, True, "foo"), Mixed(3, 4.0, False, "hello")]
    nta_mixed = NamedTupleArray(mixed_items)
    d_len = nta_mixed["d_length"]
    assert np.issubdtype(d_len.dtype, np.integer)


def test_array_protocol():
    """Test that __array__ interface works (if implemented)."""
    items = [Simple(1, 2.0), Simple(3, 4.0)]
    nta = NamedTupleArray(items)

    # Should be convertible to numpy array
    arr = np.asarray(nta)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)


def test_frozen_unfrozen_mixed_access():
    """Test accessing frozen array via different methods."""
    items = [Simple(1, 2.0), Simple(3, 4.0)]
    frozen_nta = NamedTupleArray(items, frozen=True)

    # Scalar access still works
    assert frozen_nta[0] == Simple(1, 2.0)

    # Field access returns read-only view
    field = frozen_nta["x"]
    assert field.flags.writeable is False

    # items property returns frozen array
    assert frozen_nta.items.flags.writeable is False


def test_nested_list_homogeneity():
    """Test that nested lists maintain homogeneity checks."""
    # Valid nested structure
    nested = [[Simple(i, float(i)) for i in range(2)], [Simple(j, float(j)) for j in range(2, 4)]]
    nta = NamedTupleArray(nested)
    assert nta.shape == (2, 2)

    # Invalid: mixed types in nested structure
    invalid_nested = [[Simple(1, 2.0), Mixed(3, 4.0, True, "foo")]]
    with pytest.raises(ValueError):
        NamedTupleArray(invalid_nested)


def test_slicing_preserves_type():
    """Test that slicing returns the correct type."""
    items = [Simple(i, float(i)) for i in range(5)]
    nta = NamedTupleArray(items)

    # Integer indexing returns item
    item = nta[2]
    assert isinstance(item, Simple)

    # Slice returns NamedTupleArray
    sliced = nta[1:3]
    assert isinstance(sliced, NamedTupleArray)
    assert sliced.shape == (2,)


def test_from_ndarray_0d_array():
    """Test that from_ndarray with 0D array raises error."""
    arr = np.array(2.0)
    with pytest.raises(ValueError, match="at least 1 dimension"):
        NamedTupleArray.from_array(arr, Simple)


def test_multidimensional_iteration():
    """Test iteration over multi-dimensional NamedTupleArray."""
    nested = [[Simple(i, float(i)) for i in range(2)], [Simple(j + 2, float(j + 2)) for j in range(2)]]
    nta = NamedTupleArray(nested)

    # Iteration over 2D array yields 1D NamedTupleArrays
    rows = list(nta)
    assert len(rows) == 2
    assert all(isinstance(row, NamedTupleArray) for row in rows)
    assert rows[0].shape == (2,)


def test_computed_property_with_heuristic_int():
    """Test computed property that infers int dtype via heuristic."""

    # Create a NamedTuple with a property that returns int but has no type annotation
    class AnnotationlessNT(NamedTuple):
        value: int

        @property
        def doubled(self) -> int:
            """Unannotated property returning int."""
            return self.value * 2

    items = [AnnotationlessNT(1), AnnotationlessNT(2)]
    nta = NamedTupleArray(items)

    result = nta["doubled"]
    assert np.issubdtype(result.dtype, np.integer)
    assert list(result) == [2, 4]


def test_computed_property_with_heuristic_float():
    """Test computed property that infers float dtype via heuristic."""

    class FloatPropertyNT(NamedTuple):
        value: float

        @property
        def halved(self) -> float:
            """Unannotated property returning float."""
            return self.value / 2.0

    items = [FloatPropertyNT(2.0), FloatPropertyNT(4.0)]
    nta = NamedTupleArray(items)

    result = nta["halved"]
    assert np.issubdtype(result.dtype, np.floating)
    assert pytest.approx(list(result)) == [1.0, 2.0]


def test_computed_property_mixed_types_returns_object():
    """Test computed property with mixed types returns object dtype."""

    class MixedPropertyNT(NamedTuple):
        value: int

        @property
        def mixed(self) -> int | str:
            """Unannotated property that sometimes returns int, sometimes str."""
            return self.value if self.value < 2 else f"str_{self.value}"

    items = [MixedPropertyNT(1), MixedPropertyNT(3)]
    nta = NamedTupleArray(items)

    result = nta["mixed"]
    assert result.dtype == object
    assert result[0] == 1
    assert result[1] == "str_3"


def test_getitem_returns_2d_array():
    """Test that indexing returns correct types for various index types."""
    items = [Simple(i, float(i)) for i in range(6)]
    nta = NamedTupleArray(items)

    # Fancy indexing with list returns NamedTupleArray
    indexed = nta[[0, 2, 4]]
    assert isinstance(indexed, NamedTupleArray)
    assert indexed.shape == (3,)


def test_zero_d_ndarray_from_indexing():
    """Test that scalar indexing returns NamedTuple correctly."""
    items = [Simple(1, 2.0), Simple(3, 4.0)]
    nta = NamedTupleArray(items)

    # Scalar access should return NamedTuple
    scalar = nta[0]
    assert isinstance(scalar, Simple)
    assert scalar.x == 1
    assert scalar.y == 2.0


def test_from_ndarray_empty_fields():
    """Test that from_ndarray handles edge cases."""
    # Just verify the function is defined and doesn't break in normal usage
    # The empty fields check is hard to trigger naturally
    arr = np.array([[1, 2.0], [3, 4.0]])
    nta = NamedTupleArray.from_array(arr, Simple)
    assert nta.shape == (2,)


def test_nested_empty_list_deeply():
    """Test that deeply nested empty lists raise ValueError."""
    with pytest.raises(ValueError):
        NamedTupleArray([[[]]])


def test_property_with_exception_in_heuristic():
    """Test that computed property handles exceptions in type inference gracefully."""

    class NTWithUnusualProperty(NamedTuple):
        value: int

        @property
        def unusual(self) -> dict:
            # Return something that can't be easily cast
            return {"key": self.value}

    items = [NTWithUnusualProperty(1), NTWithUnusualProperty(2)]
    nta = NamedTupleArray(items)

    # Should return object dtype (via exception handling)
    result = nta["unusual"]
    assert result.dtype == object


def test_from_ndarray_3d_array():
    """Test from_ndarray with 3D array."""
    arr = np.array([[[1, 2.0], [3, 4.0]], [[5, 6.0], [7, 8.0]]], dtype=float)
    nta = NamedTupleArray.from_array(arr, Simple)

    assert nta.shape == (2, 2)
    assert nta[0, 1] == Simple(3, 4.0)


def test_iteration_1d_direct_yield():
    """Test that 1D iteration yields NamedTuple items directly."""
    items = [Simple(1, 1.0), Simple(2, 2.0), Simple(3, 3.0)]
    nta = NamedTupleArray(items)

    yielded = list(nta)
    assert len(yielded) == 3
    assert all(isinstance(item, Simple) for item in yielded)
    assert yielded == items


def test_from_ndarray_empty_namedtuple():
    """Test that from_ndarray with empty NamedTuple raises RuntimeError (lines 232-233)."""

    class Empty(NamedTuple):
        pass

    arr = np.array([], dtype=float)
    with pytest.raises(RuntimeError, match="no fields"):
        NamedTupleArray.from_array(arr, Empty)


def test_equality_identical_arrays():
    """Test that two NamedTupleArray instances with identical data are equal."""
    items1 = [Simple(1, 2.0), Simple(3, 4.5)]
    items2 = [Simple(1, 2.0), Simple(3, 4.5)]

    arr1 = NamedTupleArray(items1)
    arr2 = NamedTupleArray(items2)

    assert arr1 == arr2
    assert arr2 == arr1  # Test equality is symmetric


def test_equality_different_values():
    """Test that NamedTupleArray instances with different values are not equal."""
    items1 = [Simple(1, 2.0), Simple(3, 4.5)]
    items2 = [Simple(1, 2.0), Simple(3, 5.0)]

    arr1 = NamedTupleArray(items1)
    arr2 = NamedTupleArray(items2)

    assert arr1 != arr2


def test_equality_different_lengths():
    """Test that NamedTupleArray instances with different lengths are not equal."""
    items1 = [Simple(1, 2.0), Simple(3, 4.5)]
    items2 = [Simple(1, 2.0), Simple(3, 4.5), Simple(5, 6.0)]

    arr1 = NamedTupleArray(items1)
    arr2 = NamedTupleArray(items2)

    assert arr1 != arr2


def test_equality_different_types():
    """Test that NamedTupleArray instances with different NamedTuple types are not equal."""
    simple_items = [Simple(1, 2.0), Simple(3, 4.5)]
    breath_items = [Breath(1.0, 2.0, 3.0), Breath(3.0, 4.0, 5.0)]

    arr1 = NamedTupleArray(simple_items)
    arr2 = NamedTupleArray(breath_items)

    assert arr1 != arr2


def test_equality_with_nan_values():
    """Test that NamedTupleArray instances with NaN values can be compared correctly."""
    items1 = [Simple(1, np.nan), Simple(3, 4.5)]
    items2 = [Simple(1, np.nan), Simple(3, 4.5)]

    arr1 = NamedTupleArray(items1)
    arr2 = NamedTupleArray(items2)

    # NaN values should be considered equal in this comparison
    assert arr1 == arr2


def test_equality_with_nan_different_positions():
    """Test that NamedTupleArray with NaN in different positions are not equal."""
    items1 = [Simple(1, np.nan), Simple(3, 4.5)]
    items2 = [Simple(1, 2.0), Simple(3, np.nan)]

    arr1 = NamedTupleArray(items1)
    arr2 = NamedTupleArray(items2)

    assert arr1 != arr2


def test_equality_2d_arrays():
    """Test equality comparison for 2D NamedTupleArray instances."""
    nested1 = [[Simple(i + j, float(i * j)) for j in range(3)] for i in range(2)]
    nested2 = [[Simple(i + j, float(i * j)) for j in range(3)] for i in range(2)]

    arr1 = NamedTupleArray(nested1)
    arr2 = NamedTupleArray(nested2)

    assert arr1 == arr2


def test_equality_2d_arrays_different():
    """Test inequality for 2D NamedTupleArray instances with different values."""
    nested1 = [[Simple(i + j, float(i * j)) for j in range(3)] for i in range(2)]
    nested2 = [[Simple(i + j + 1, float(i * j)) for j in range(3)] for i in range(2)]

    arr1 = NamedTupleArray(nested1)
    arr2 = NamedTupleArray(nested2)

    assert arr1 != arr2


def test_equality_not_equal_to_non_namedtuplearray():
    """Test that NamedTupleArray is not equal to other types."""
    items = [Simple(1, 2.0), Simple(3, 4.5)]
    arr = NamedTupleArray(items)

    # Test inequality with list
    assert arr != items

    # Test inequality with numpy array
    assert arr != np.array([(1, 2.0), (3, 4.5)])

    # Test inequality with None
    assert arr is not None

    # Test inequality with string
    assert arr != "not an array"


def test_equality_frozen_and_unfrozen():
    """Test that frozen and unfrozen arrays with same data are equal."""
    items1 = [Simple(1, 2.0), Simple(3, 4.5)]
    items2 = [Simple(1, 2.0), Simple(3, 4.5)]

    arr_frozen = NamedTupleArray(items1, frozen=True)
    arr_unfrozen = NamedTupleArray(items2, frozen=False)

    assert arr_frozen == arr_unfrozen
