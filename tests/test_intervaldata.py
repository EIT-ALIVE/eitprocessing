import random

import numpy as np
import pytest

from eitprocessing.datahandling.intervaldata import Interval, IntervalData


@pytest.fixture()
def intervaldata1():
    """Creates an IntervalData object with n sequential intervals lasting 1 second each."""
    n = random.randint(50, 150)
    return IntervalData(
        label="intervaldata1",
        name="IntervalData 1",
        unit=None,
        category="dummy",
        intervals=list(zip(range(n), range(1, n + 1), strict=True)),
        default_partial_inclusion=True,
    )


@pytest.fixture()
def intervaldata2():
    """Creates an IntervalData object with n sequential intervals lasting 1 second each."""
    n = random.randint(50, 150)
    return IntervalData(
        label="intervaldata1",
        name="IntervalData 2",
        unit=None,
        category="dummy",
        intervals=list(zip(range(n), range(1, n + 1), strict=True)),
        default_partial_inclusion=False,
    )


@pytest.fixture()
def intervaldata3():
    """Creates an IntervalData object with n sequential intervals lasting 1 second each, with values."""
    n = random.randint(50, 150)
    return IntervalData(
        label="intervaldata1",
        name="IntervalData 2",
        unit=None,
        category="dummy",
        intervals=list(zip(range(n), range(1, n + 1), strict=True)),
        default_partial_inclusion=False,
        values=[random.random() for _ in range(n)],
    )


@pytest.fixture()
def intervaldata4():
    """Creates an IntervalData object with n sequential intervals lasting 1 second each, with values."""
    n = random.randint(50, 150)
    return IntervalData(
        label="intervaldata1",
        name="IntervalData 2",
        unit=None,
        category="dummy",
        intervals=list(zip(range(n), range(1, n + 1), strict=True)),
        default_partial_inclusion=False,
        values=np.array([random.random() for _ in range(n)]),
    )


def test_post_init(intervaldata1: IntervalData) -> None:
    assert isinstance(intervaldata1.intervals, list)
    assert all(isinstance(interval, Interval) for interval in intervaldata1.intervals)


def test_len(intervaldata1: IntervalData) -> None:
    assert len(intervaldata1) == len(intervaldata1.intervals)


def test_has_values(
    intervaldata1: IntervalData,
    intervaldata2: IntervalData,
    intervaldata3: IntervalData,
    intervaldata4: IntervalData,
) -> None:
    assert not intervaldata1.has_values
    intervaldata1.values = []
    assert intervaldata1.has_values
    intervaldata1.values = None

    assert not intervaldata2.has_values
    assert intervaldata3.has_values
    assert intervaldata4.has_values


def test_index_slicing(intervaldata1: IntervalData) -> None:
    _sliced_copy = intervaldata1._sliced_copy(0, 10, newlabel="sliced_copy")  # noqa: SLF001
    assert len(_sliced_copy) == 10
    sliced_copy = intervaldata1[:10]
    assert _sliced_copy == sliced_copy

    assert len(intervaldata1[0]) == 1


def test_select_by_time(intervaldata1: IntervalData, intervaldata2: IntervalData) -> None:
    assert intervaldata1.t[:1] == intervaldata1[:1]

    assert len(intervaldata1.t[:1]) == 1
    assert len(intervaldata1.t[:0.5]) == 1

    assert len(intervaldata2.t[:1]) == 1
    assert len(intervaldata2.t[:0.5]) == 0

    assert len(intervaldata1.t[1:2]) == 1
    assert len(intervaldata2.t[1:2]) == 1

    assert len(intervaldata1.t[1.5:2.5]) == 2
    assert len(intervaldata2.t[1.5:2.5]) == 0

    assert len(intervaldata1.t[1.5:3]) == 2
    assert len(intervaldata2.t[1.5:3]) == 1

    assert len(intervaldata1.t[1.5:3.5]) == 3
    assert len(intervaldata2.t[1.5:3.5]) == 1

    assert len(intervaldata1.t[2:3.5]) == 2
    assert len(intervaldata2.t[2:3.5]) == 1

    sliced_copy = intervaldata1.t[2.5:3.5]
    assert sliced_copy.intervals[0].start_time == 2.5
    assert sliced_copy.intervals[0].end_time == 3
    assert sliced_copy.intervals[1].start_time == 3
    assert sliced_copy.intervals[1].end_time == 3.5

    sliced_copy = intervaldata1.t[-10:-1]
    assert len(sliced_copy) == 0

    sliced_copy = intervaldata1.t[-10:3]
    assert len(sliced_copy) == 3

    sliced_copy = intervaldata1.t[:]
    assert len(sliced_copy) == len(intervaldata1)

    assert intervaldata1.t[:10] == intervaldata1.t[0:10]
    assert intervaldata1.t[20:] == intervaldata1.t[20 : len(intervaldata1) + 1]


def test_select_by_time_values(intervaldata3: IntervalData):
    assert isinstance(intervaldata3.values, list)

    sliced_copy = intervaldata3[:10]
    assert len(sliced_copy.intervals) == len(sliced_copy.values)
    assert sliced_copy.values == intervaldata3.values[:10]


def test_concatenate(intervaldata1: IntervalData) -> None:
    sliced_copy_1 = intervaldata1[:10]
    sliced_copy_2 = intervaldata1[10:20]

    assert len(sliced_copy_1) == 10
    assert len(sliced_copy_2) == 10

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated) == 20
    assert concatenated == sliced_copy_1.concatenate(sliced_copy_2)

    assert sliced_copy_1.intervals == concatenated.intervals[:10]
    assert sliced_copy_2.intervals == concatenated.intervals[10:]

    sliced_copy_3 = intervaldata1[1000:]
    assert sliced_copy_1 + sliced_copy_3 == sliced_copy_1
    assert sliced_copy_3 + sliced_copy_1 == sliced_copy_1

    with pytest.raises(ValueError):
        sliced_copy_2 + sliced_copy_1


def test_concatenate_values_list(intervaldata3: IntervalData):
    sliced_copy_1 = intervaldata3[:10]
    sliced_copy_2 = intervaldata3[10:20]

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated.intervals) == len(concatenated.values)
    assert isinstance(intervaldata3.values, list)
    assert concatenated.values == intervaldata3.values[:20]


def test_concatenate_values_numpy(intervaldata4: IntervalData) -> None:
    sliced_copy_1 = intervaldata4[:10]
    sliced_copy_2 = intervaldata4[10:20]

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated.intervals) == len(concatenated.values)
    assert isinstance(intervaldata4.values, np.ndarray)
    assert np.array_equal(concatenated.values, intervaldata4.values[:20])


def test_concatenate_values_type_mismatch(intervaldata3: IntervalData, intervaldata4: IntervalData) -> None:
    with pytest.raises(TypeError):
        intervaldata3[:10] + intervaldata4[10:]
