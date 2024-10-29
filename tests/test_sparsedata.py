import bisect
import random

import numpy as np
import pytest

from eitprocessing.datahandling.sparsedata import SparseData


@pytest.fixture
def sparsedata_novalues():
    """SparseData object without values and random time points."""
    n = random.randint(50, 150)
    return SparseData(
        label="sparsedata_novalues",
        name="SparseData without values",
        unit=None,
        category="dummy",
        time=np.array(sorted({random.randint(0, 1000) for _ in range(n)})),
    )


@pytest.fixture
def sparsedata_valueslist():
    """SparseData object with random values as list at random time points."""
    n = random.randint(50, 150)
    time = np.array(sorted({random.randint(0, 1000) for _ in range(n)}))
    values = [random.random() for _ in range(len(time))]
    return SparseData(
        label="sparsedata_valueslist",
        name="SparseData with values as list",
        unit=None,
        category="dummy",
        time=time,
        values=values,
    )


@pytest.fixture
def sparsedata_valuesarray():
    """SparseData object with random values as array at random time points."""
    n = random.randint(50, 150)
    time = np.array(sorted({random.randint(0, 1000) for _ in range(n)}))
    values = np.array([random.random() for _ in range(len(time))])
    return SparseData(
        label="sparsedata_valuesarray",
        name="SparseData with values as array",
        unit=None,
        category="dummy",
        time=time,
        values=values,
    )


def test_len(sparsedata_novalues: SparseData) -> None:
    assert len(sparsedata_novalues) == len(sparsedata_novalues.time)


def test_has_values(
    sparsedata_novalues: SparseData,
    sparsedata_valueslist: SparseData,
    sparsedata_valuesarray: SparseData,
) -> None:
    assert not sparsedata_novalues.has_values
    sparsedata_novalues.values = []
    assert sparsedata_novalues.has_values
    sparsedata_novalues.values = None

    assert sparsedata_valueslist.has_values
    assert sparsedata_valuesarray.has_values


def test_index_slicing(sparsedata_novalues: SparseData) -> None:
    _sliced_copy = sparsedata_novalues._sliced_copy(0, 10, newlabel="sliced_copy")
    assert len(_sliced_copy) == 10
    sliced_copy = sparsedata_novalues[:10]
    assert _sliced_copy == sliced_copy

    assert len(sparsedata_novalues[0]) == 1


def test_select_by_time() -> None:
    pytest.skip("This should be filled after finishing selection by time")


def test_select_by_time_values() -> None:
    pytest.skip("This should be filled after finishing selection by time")


def test_concatenate(sparsedata_novalues: SparseData) -> None:
    sliced_copy_1 = sparsedata_novalues[:10]
    sliced_copy_2 = sparsedata_novalues[10:20]

    assert len(sliced_copy_1) == 10
    assert len(sliced_copy_2) == 10

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated) == 20
    assert concatenated == sliced_copy_1.concatenate(sliced_copy_2)

    assert np.array_equal(sliced_copy_1.time, concatenated.time[:10])
    assert np.array_equal(sliced_copy_2.time, concatenated.time[10:])

    sliced_copy_3 = sparsedata_novalues[1000:]
    assert sliced_copy_1 + sliced_copy_3 == sliced_copy_1
    assert sliced_copy_3 + sliced_copy_1 == sliced_copy_1

    with pytest.raises(ValueError):
        sliced_copy_2 + sliced_copy_1


def test_concatenate_values_list(sparsedata_valueslist: SparseData):
    sliced_copy_1 = sparsedata_valueslist[:10]
    sliced_copy_2 = sparsedata_valueslist[10:20]

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated.time) == len(concatenated.values)
    assert isinstance(sparsedata_valueslist.values, list)
    assert concatenated.values == sparsedata_valueslist.values[:20]


def test_concatenate_values_array(sparsedata_valuesarray: SparseData):
    sliced_copy_1 = sparsedata_valuesarray[:10]
    sliced_copy_2 = sparsedata_valuesarray[10:20]

    concatenated = sliced_copy_1 + sliced_copy_2
    assert len(concatenated.time) == len(concatenated.values)
    assert isinstance(sparsedata_valuesarray.values, np.ndarray)
    assert np.array_equal(concatenated.values, sparsedata_valuesarray.values[:20])


def test_concatenate_values_mismatch(sparsedata_valueslist: SparseData, sparsedata_valuesarray: SparseData) -> None:
    first_end_index = 10
    first_end_time = sparsedata_valueslist.time[first_end_index]
    second_start_index = bisect.bisect_left(sparsedata_valuesarray.time, first_end_time)
    with pytest.raises(TypeError):
        sparsedata_valueslist[:first_end_index] + sparsedata_valuesarray[second_start_index:]
