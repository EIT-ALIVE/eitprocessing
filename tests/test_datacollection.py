from collections import UserDict
from collections.abc import Callable

import numpy as np
import pytest

from eitprocessing.datahandling import DataContainer
from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.sparsedata import SparseData


@pytest.fixture
def create_data_object() -> Callable[[str, list | None], ContinuousData]:
    def internal(
        label: str,
        derived_from: list | None = None,
        time: np.ndarray | None = None,
        values: np.ndarray | None = None,
    ) -> ContinuousData:
        return ContinuousData(
            label=label,
            name="",
            unit="",
            category="physical measurements",
            time=time if isinstance(time, np.ndarray) else np.array([]),
            sample_frequency=0,
            values=values if isinstance(values, np.ndarray) else np.array([]),
            derived_from=derived_from if isinstance(derived_from, list) else [],
        )

    return internal


def test_init():
    _ = DataCollection(ContinuousData)
    _ = DataCollection(IntervalData)
    _ = DataCollection(EITData)
    _ = DataCollection(SparseData)

    with pytest.raises(TypeError):
        _ = DataCollection(str)

    with pytest.raises(TypeError):
        _ = DataCollection(DataContainer)


def test_check_item(create_data_object: Callable[[str], ContinuousData]):
    data_object = create_data_object(label="test label")
    data_object_b = create_data_object(label="test label")

    dc = DataCollection(ContinuousData)

    dc._check_item(data_object, key="test label")
    dc._check_item(data_object)

    with pytest.raises(KeyError):
        # key does not match label
        dc._check_item(data_object, key="not test label")

    with pytest.raises(TypeError):
        # object type does not match
        dc._check_item("string")

    # do not use dc.__setitem__() (dc[...] = ...) which uses dc._check_item()
    UserDict.__setitem__(dc, "test label", data_object)

    assert "test label" in dc

    with pytest.raises(KeyError):
        # do not allow overwriting with the same label
        dc._check_item(data_object_b)

    dc._check_item(data_object_b, overwrite=True)


def test_check_add(create_data_object: Callable[[str], ContinuousData]):
    data_object_1 = create_data_object("label 1")
    data_object_1_b = create_data_object("label 1")
    data_object_2 = create_data_object("label 2")
    data_object_3 = create_data_object("label 3")

    dc = DataCollection(ContinuousData)
    assert len(dc) == 0

    dc.add(data_object_1)
    assert "label 1" in dc
    assert len(dc) == 1

    dc.add(data_object_2, data_object_3)
    assert "label 2" in dc
    assert "label 3" in dc
    assert len(dc) == 3

    with pytest.raises(KeyError):
        dc.add(data_object_1_b)

    assert dc["label 1"] is not data_object_1_b
    dc.add(data_object_1_b, overwrite=True)
    assert dc["label 1"] is data_object_1_b
    assert len(dc) == 3


def test_set_item(create_data_object: Callable[[str], ContinuousData]):
    data_object_1 = create_data_object("label 1")
    data_object_1_b = create_data_object("label 1")

    dc = DataCollection(ContinuousData)
    dc["label 1"] = data_object_1

    with pytest.raises(KeyError, match=r"(.*?) does not match label (.*?)"):
        dc["label 2"] = data_object_1

    with pytest.raises(KeyError, match=r"Item with label (.*?) already exists"):
        # duplicate keys
        dc["label 1"] = data_object_1_b


def test_loaded_derived_data(create_data_object: Callable):
    data_object_loaded_1 = create_data_object("label 1")
    data_object_loaded_2 = create_data_object("label 2")
    data_object_derived_1_a = create_data_object("label 1 der a", derived_from=[data_object_loaded_1])
    data_object_derived_1_b = create_data_object("label 1 der b", derived_from=[data_object_loaded_1])
    data_object_derived_2_a = create_data_object("label 2 der a", derived_from=[data_object_loaded_2])

    dc = DataCollection(ContinuousData)
    dc.add(
        data_object_loaded_1,
        data_object_loaded_2,
        data_object_derived_1_a,
        data_object_derived_1_b,
        data_object_derived_2_a,
    )

    assert dc.get_loaded_data() == {
        "label 1": data_object_loaded_1,
        "label 2": data_object_loaded_2,
    }
    assert dc.get_derived_data() == {
        "label 1 der a": data_object_derived_1_a,
        "label 1 der b": data_object_derived_1_b,
        "label 2 der a": data_object_derived_2_a,
    }
    assert dc.get_data_derived_from(data_object_loaded_1) == {
        "label 1 der a": data_object_derived_1_a,
        "label 1 der b": data_object_derived_1_b,
    }
    assert dc.get_data_derived_from(data_object_loaded_2) == {
        "label 2 der a": data_object_derived_2_a,
    }


def test_concatenate(create_data_object: Callable):
    time = np.arange(100) / 20
    rng = np.random.default_rng()
    values = rng.random(time.shape)

    data_object_1_a = create_data_object("label 1", time=time[:30], values=values[:30])
    data_object_1_b = create_data_object("label 1", time=time[30:], values=values[30:])
    data_object_1_c = create_data_object("label 1", time=time[28:], values=values[28:])
    data_object_2 = create_data_object("label 2", time=time[30:], values=values[30:])

    dc_1_a = DataCollection(ContinuousData)
    dc_1_b = DataCollection(ContinuousData)
    dc_1_c = DataCollection(ContinuousData)
    dc_2 = DataCollection(ContinuousData)

    dc_1_a.add(data_object_1_a)
    dc_1_b.add(data_object_1_b)
    dc_1_c.add(data_object_1_c)
    dc_2.add(data_object_2)

    dc_1_concat = dc_1_a.concatenate(dc_1_b)
    assert np.array_equal(dc_1_concat["label 1"].time, time)
    assert np.array_equal(dc_1_concat["label 1"].values, values)

    with pytest.raises(ValueError, match="Keys don't match"):
        dc_1_a.concatenate(dc_2)

    with pytest.raises(ValueError, match=r"(.*?) \(b\) starts before (.*?) \(a\) end"):
        dc_1_b.concatenate(dc_1_a)

    with pytest.raises(ValueError, match=r"(.*?) \(b\) starts before (.*?) \(a\) end"):
        dc_1_a.concatenate(dc_1_c)


def test_select_by_time(create_data_object: Callable):
    time1 = np.arange(0, 100)
    time2 = np.arange(50, 150)
    rng = np.random.default_rng()
    values1 = rng.random(time1.shape)
    values2 = rng.random(time2.shape)

    data_object_1 = create_data_object("label 1", time=time1, values=values1)
    data_object_2 = create_data_object("label 2", time=time2, values=values2)

    dc = DataCollection(ContinuousData)
    dc.add(data_object_1, data_object_2)

    s1 = dc.select_by_time(20, 80)
    # TODO Add test with start == end

    assert np.array_equal(s1["label 1"].time, time1[20:80])
    assert np.array_equal(s1["label 1"].values, values1[20:80])
    assert np.array_equal(s1["label 2"].time, time2[:30])
    assert np.array_equal(s1["label 2"].values, values2[:30])
