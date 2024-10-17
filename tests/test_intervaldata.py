import random

import numpy as np
import pytest

from eitprocessing.datahandling.intervaldata import Interval, IntervalData


@pytest.fixture
def intervaldata_novalues_partialtrue():
    """IntervalData object without values, and partial inclusion set to True by default.

    Creates an IntervalData object with n sequential intervals lasting 1 second each.
    """
    n = 50
    return IntervalData(
        label="intervaldata_novalues_partialtrue",
        name="IntervalData without values, with partial inclusion",
        unit=None,
        category="dummy",
        intervals=list(zip(range(n), range(1, n + 1), strict=True)),
        default_partial_inclusion=True,
    )


@pytest.fixture
def intervaldata_novalues_partialfalse():
    """IntervalData object without values, without partial inclusion.

    Creates an IntervalData object with n sequential intervals lasting 1 second each.
    """
    n = 56
    return IntervalData(
        label="intervaldata_novalues_partialfalse",
        name="IntervalData without values, without partial inclusion",
        unit=None,
        category="dummy",
        intervals=list(zip(range(n), range(1, n + 1), strict=True)),
        default_partial_inclusion=False,
    )


@pytest.fixture
def intervaldata_valueslist_partialfalse():
    """IntervalData object with values as list, no partial inclusion.

    Creates an IntervalData object with n sequential intervals lasting 1 second each, with values.
    """
    n = 62
    return IntervalData(
        label="intervaldata_listvalues_partialfalse",
        name="IntervalData with values as list, no partial inclusion",
        unit=None,
        category="dummy",
        intervals=list(zip(range(n), range(1, n + 1), strict=True)),
        default_partial_inclusion=False,
        values=[random.random() for _ in range(n)],
    )


@pytest.fixture
def intervaldata_valuesarray_partialfalse():
    """IntervalData object with values as numpy array, no partial inclusion.

    Creates an IntervalData object with n sequential intervals lasting 1 second each, with values.
    """
    n = 68
    return IntervalData(
        label="intervaldata_valuesarray_partialfalse",
        name="IntervalData with values as array, no partial inclusion",
        unit=None,
        category="dummy",
        intervals=list(zip(range(n), range(1, n + 1), strict=True)),
        default_partial_inclusion=False,
        values=np.array([random.random() for _ in range(n)]),
    )


def test_post_init(intervaldata_novalues_partialtrue: IntervalData):
    assert isinstance(intervaldata_novalues_partialtrue.intervals, list)
    assert all(isinstance(interval, Interval) for interval in intervaldata_novalues_partialtrue.intervals)


def test_len(intervaldata_novalues_partialtrue: IntervalData):
    assert len(intervaldata_novalues_partialtrue) == len(intervaldata_novalues_partialtrue.intervals)


def test_has_values(
    intervaldata_novalues_partialtrue: IntervalData,
    intervaldata_novalues_partialfalse: IntervalData,
    intervaldata_valueslist_partialfalse: IntervalData,
    intervaldata_valuesarray_partialfalse: IntervalData,
) -> None:
    assert not intervaldata_novalues_partialtrue.has_values
    intervaldata_novalues_partialtrue.values = []
    assert intervaldata_novalues_partialtrue.has_values
    intervaldata_novalues_partialtrue.values = None
    assert not intervaldata_novalues_partialtrue.has_values

    assert not intervaldata_novalues_partialfalse.has_values
    assert intervaldata_valueslist_partialfalse.has_values
    assert intervaldata_valuesarray_partialfalse.has_values


def test_index_slicing(intervaldata_novalues_partialtrue: IntervalData):
    _sliced_copy = intervaldata_novalues_partialtrue._sliced_copy(0, 10, newlabel="sliced_copy")
    assert len(_sliced_copy) == 10
    sliced_copy = intervaldata_novalues_partialtrue[:10]
    assert _sliced_copy == sliced_copy

    assert len(intervaldata_novalues_partialtrue[0]) == 1


def test_select_by_time(
    intervaldata_novalues_partialtrue: IntervalData,
    intervaldata_novalues_partialfalse: IntervalData,
):
    assert intervaldata_novalues_partialtrue.t[:1] == intervaldata_novalues_partialtrue[:1]

    assert len(intervaldata_novalues_partialtrue.t[:1]) == 1
    assert len(intervaldata_novalues_partialfalse.t[:1]) == 1

    assert len(intervaldata_novalues_partialtrue.t[:0.5]) == 1
    assert len(intervaldata_novalues_partialfalse.t[:0.5]) == 0

    assert len(intervaldata_novalues_partialtrue.t[1:2]) == 1
    assert len(intervaldata_novalues_partialfalse.t[1:2]) == 1

    assert len(intervaldata_novalues_partialtrue.t[1.5:2.5]) == 2
    assert len(intervaldata_novalues_partialfalse.t[1.5:2.5]) == 0

    assert len(intervaldata_novalues_partialtrue.t[1.5:3]) == 2
    assert len(intervaldata_novalues_partialfalse.t[1.5:3]) == 1

    assert len(intervaldata_novalues_partialtrue.t[1.5:3.5]) == 3
    assert len(intervaldata_novalues_partialfalse.t[1.5:3.5]) == 1

    assert len(intervaldata_novalues_partialtrue.t[2:3.5]) == 2
    assert len(intervaldata_novalues_partialfalse.t[2:3.5]) == 1

    sliced_copy = intervaldata_novalues_partialtrue.t[2.5:3.5]
    assert sliced_copy.intervals[0].start_time == 2.5
    assert sliced_copy.intervals[0].end_time == 3
    assert sliced_copy.intervals[1].start_time == 3
    assert sliced_copy.intervals[1].end_time == 3.5

    sliced_copy = intervaldata_novalues_partialtrue.t[-10:-1]
    assert len(sliced_copy) == 0

    sliced_copy = intervaldata_novalues_partialtrue.t[-10:3]
    assert len(sliced_copy) == 3

    sliced_copy = intervaldata_novalues_partialtrue.t[:]
    assert sliced_copy == intervaldata_novalues_partialtrue
    assert sliced_copy is not intervaldata_novalues_partialtrue

    assert intervaldata_novalues_partialtrue.t[:10] == intervaldata_novalues_partialtrue.t[0:10]
    assert (
        intervaldata_novalues_partialtrue.t[20:]
        == intervaldata_novalues_partialtrue.t[20 : len(intervaldata_novalues_partialtrue) + 1]
    )


def test_select_by_time_values(intervaldata_valueslist_partialfalse: IntervalData):
    assert isinstance(intervaldata_valueslist_partialfalse.values, list)

    sliced_copy = intervaldata_valueslist_partialfalse[:10]
    assert len(sliced_copy.intervals) == len(sliced_copy.values)
    assert sliced_copy.values == intervaldata_valueslist_partialfalse.values[:10]


def test_concatenate(intervaldata_novalues_partialtrue: IntervalData):
    sliced_copy_1 = intervaldata_novalues_partialtrue[:10]
    sliced_copy_2 = intervaldata_novalues_partialtrue[10:20]

    assert len(sliced_copy_1) == 10
    assert len(sliced_copy_2) == 10

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated) == 20
    assert concatenated == sliced_copy_1.concatenate(sliced_copy_2)

    assert sliced_copy_1.intervals == concatenated.intervals[:10]
    assert sliced_copy_2.intervals == concatenated.intervals[10:]

    sliced_copy_3 = intervaldata_novalues_partialtrue[1000:]
    assert sliced_copy_1 + sliced_copy_3 == sliced_copy_1
    assert sliced_copy_3 + sliced_copy_1 == sliced_copy_1

    with pytest.raises(ValueError):
        sliced_copy_2 + sliced_copy_1


def test_concatenate_values_list(intervaldata_valueslist_partialfalse: IntervalData):
    sliced_copy_1 = intervaldata_valueslist_partialfalse[:10]
    sliced_copy_2 = intervaldata_valueslist_partialfalse[10:20]

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated.intervals) == len(concatenated.values)
    assert isinstance(intervaldata_valueslist_partialfalse.values, list)
    assert concatenated.values == intervaldata_valueslist_partialfalse.values[:20]


def test_concatenate_values_numpy(intervaldata_valuesarray_partialfalse: IntervalData):
    sliced_copy_1 = intervaldata_valuesarray_partialfalse[:10]
    sliced_copy_2 = intervaldata_valuesarray_partialfalse[10:20]

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated.intervals) == len(concatenated.values)
    assert isinstance(intervaldata_valuesarray_partialfalse.values, np.ndarray)
    assert np.array_equal(concatenated.values, intervaldata_valuesarray_partialfalse.values[:20])


def test_concatenate_values_type_mismatch(
    intervaldata_valueslist_partialfalse: IntervalData,
    intervaldata_valuesarray_partialfalse: IntervalData,
):
    with pytest.raises(TypeError):
        intervaldata_valueslist_partialfalse[:10] + intervaldata_valuesarray_partialfalse[10:]
