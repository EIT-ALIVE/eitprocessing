import bisect
import random

import numpy as np
import pytest

from eitprocessing.datahandling.sparsedata import SparseData


@pytest.fixture()
def sparsedata1():
    """Creates an SparseData object with n sequential intervals lasting 1 second each."""
    n = random.randint(50, 150)
    return SparseData(
        label="sparsedata1",
        name="SparseData 1",
        unit=None,
        category="dummy",
        time=np.array(sorted({random.randint(0, 1000) for _ in range(n)})),
    )


@pytest.fixture()
def sparsedata2():
    """Creates an SparseData object with n sequential intervals lasting 1 second each."""
    n = random.randint(50, 150)
    return SparseData(
        label="sparsedata1",
        name="SparseData 2",
        unit=None,
        category="dummy",
        time=np.array(sorted({random.randint(0, 1000) for _ in range(n)})),
    )


@pytest.fixture()
def sparsedata3():
    """Creates an SparseData object with n sequential intervals lasting 1 second each, with values."""
    n = random.randint(50, 150)
    return SparseData(
        label="sparsedata1",
        name="SparseData 2",
        unit=None,
        category="dummy",
        time=np.array(sorted({random.randint(0, 1000) for _ in range(n)})),
        values=[random.random() for _ in range(n)],
    )


@pytest.fixture()
def sparsedata4():
    """Creates an SparseData object with n sequential intervals lasting 1 second each, with values."""
    n = random.randint(50, 150)
    return SparseData(
        label="sparsedata1",
        name="SparseData 2",
        unit=None,
        category="dummy",
        time=np.array(sorted({random.randint(0, 1000) for _ in range(n)})),
        values=np.array([random.random() for _ in range(n)]),
    )


def test_len(sparsedata1: SparseData) -> None:
    assert len(sparsedata1) == len(sparsedata1.time)


def test_has_values(
    sparsedata1: SparseData,
    sparsedata2: SparseData,
    sparsedata3: SparseData,
    sparsedata4: SparseData,
) -> None:
    assert not sparsedata1.has_values
    sparsedata1.values = []
    assert sparsedata1.has_values
    sparsedata1.values = None

    assert not sparsedata2.has_values
    assert sparsedata3.has_values
    assert sparsedata4.has_values


def test_index_slicing(sparsedata1: SparseData) -> None:
    _sliced_copy = sparsedata1._sliced_copy(0, 10, newlabel="sliced_copy")  # noqa: SLF001
    assert len(_sliced_copy) == 10
    sliced_copy = sparsedata1[:10]
    assert _sliced_copy == sliced_copy

    assert len(sparsedata1[0]) == 1


def test_select_by_time() -> None:
    pytest.skip("This should be filled after finishing selection by time")


def test_select_by_time_values() -> None:
    pytest.skip("This should be filled after finishing selection by time")


def test_concatenate(sparsedata1: SparseData) -> None:
    sliced_copy_1 = sparsedata1[:10]
    sliced_copy_2 = sparsedata1[10:20]

    assert len(sliced_copy_1) == 10
    assert len(sliced_copy_2) == 10

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated) == 20
    assert concatenated == sliced_copy_1.concatenate(sliced_copy_2)

    assert np.array_equal(sliced_copy_1.time, concatenated.time[:10])
    assert np.array_equal(sliced_copy_2.time, concatenated.time[10:])

    sliced_copy_3 = sparsedata1[1000:]
    assert sliced_copy_1 + sliced_copy_3 == sliced_copy_1
    assert sliced_copy_3 + sliced_copy_1 == sliced_copy_1

    with pytest.raises(ValueError):
        sliced_copy_2 + sliced_copy_1


def test_concatenate_values_list(sparsedata3: SparseData):
    sliced_copy_1 = sparsedata3[:10]
    sliced_copy_2 = sparsedata3[10:20]

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated.time) == len(concatenated.values)
    assert isinstance(sparsedata3.values, list)
    assert concatenated.values == sparsedata3.values[:20]


def test_concatenate_values_array(sparsedata4: SparseData):
    sliced_copy_1 = sparsedata4[:10]
    sliced_copy_2 = sparsedata4[10:20]

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated.time) == len(concatenated.values)
    assert isinstance(sparsedata4.values, np.ndarray)
    assert np.array_equal(concatenated.values, sparsedata4.values[:20])


def test_concatenate_values_mismatch(sparsedata3: SparseData, sparsedata4: SparseData) -> None:
    first_end_index = 10
    first_end_time = sparsedata3.time[first_end_index]
    second_start_index = bisect.bisect_left(sparsedata4.time, first_end_time)
    with pytest.raises(TypeError):
        sparsedata3[:first_end_index] + sparsedata4[second_start_index:]
