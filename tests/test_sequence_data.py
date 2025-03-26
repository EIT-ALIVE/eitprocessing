from collections.abc import Callable

import numpy as np
import pytest

from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.sequence import Sequence, _DataAccess
from tests.test_breath_detection import ContinuousData


@pytest.fixture
def create_continuous_data_object():
    return lambda label: ContinuousData(
        label=label,
        name="name",
        unit="unit",
        category="other",
        time=np.array([]),
        values=np.array([]),
    )


@pytest.fixture
def create_interval_data_object():
    return lambda label: IntervalData(
        label=label,
        name="name",
        unit="unit",
        category="other",
        intervals=[],
        values=[],
    )


def test_init():
    sequence = Sequence()

    assert isinstance(sequence.data, _DataAccess)
    assert sequence.data._collections == (
        sequence.continuous_data,
        sequence.interval_data,
        sequence.sparse_data,
        sequence.eit_data,
    )
    assert hasattr(sequence.data, "get")
    assert hasattr(sequence.data, "add")


def test_get(create_continuous_data_object: Callable, create_interval_data_object: Callable):
    sequence = Sequence()

    assert sequence.data.get("foo", None) is None
    assert "foo" not in sequence.data

    with pytest.raises(KeyError):
        sequence.data.get("foo")

    with pytest.raises(KeyError):
        sequence.data["foo"]

    continuous_data_object = create_continuous_data_object("foo")
    sequence.continuous_data.add(continuous_data_object)

    assert continuous_data_object in sequence.continuous_data.values()
    assert sequence.data.get("foo", None) is continuous_data_object
    assert "foo" in sequence.data
    assert sequence.data.get("foo") is continuous_data_object
    assert sequence.data["foo"] is continuous_data_object

    interval_data_object = create_interval_data_object("bar")
    sequence.interval_data.add(interval_data_object)

    assert interval_data_object in sequence.interval_data.values()
    assert sequence.data.get("bar", None) is interval_data_object
    assert "bar" in sequence.data
    assert sequence.data.get("bar") is interval_data_object
    assert sequence.data["bar"] is interval_data_object


def test_add(create_continuous_data_object: Callable, create_interval_data_object: Callable):
    sequence = Sequence()

    continuous_data_object = create_continuous_data_object("foo")
    sequence.data.add(continuous_data_object)

    assert "foo" in sequence.data
    assert continuous_data_object in sequence.continuous_data.values()

    interval_data_object = create_interval_data_object("bar")
    sequence.data["bar"] = interval_data_object

    assert "bar" in sequence.data
    assert interval_data_object in sequence.interval_data.values()

    continuous_data_object_2 = create_continuous_data_object("foobar")
    with pytest.raises(KeyError):
        sequence.data["not foobar"] = continuous_data_object_2


def test_add_multiple(create_continuous_data_object: Callable, create_interval_data_object: Callable):
    sequence = Sequence()

    continuous_data_object = create_continuous_data_object("foo")
    interval_data_object = create_interval_data_object("bar")

    sequence.data.add(continuous_data_object, interval_data_object)
    assert "foo" in sequence.data
    assert "bar" in sequence.data


def test_duplicate_keys(
    create_continuous_data_object: Callable, create_interval_data_object: Callable
):
    sequence = Sequence()

    continuous_data_object = create_continuous_data_object("foo")
    sequence.continuous_data.add(continuous_data_object)

    assert "foo" in sequence.continuous_data

    interval_data_object = create_interval_data_object("foo")

    with pytest.raises(KeyError):
        sequence.data.add(interval_data_object)

    # you can still add through the DataCollection, but this interface will not be available anymore
    sequence.interval_data.add(interval_data_object)

    with pytest.raises(KeyError):
        _ = sequence.data
