import pytest

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.sequence import Sequence


def test_lock_continuousdata_values(draeger1: Sequence):
    assert isinstance(draeger1, Sequence)

    try:
        cd = next(iter(draeger1.continuous_data.values()))
    except:
        pytest.fail("No continuous data available in sequence")

    assert isinstance(cd, ContinuousData)
    assert cd.is_lockable("values")
    assert cd.is_locked("values")
    try:
        cd.unlock("values")
    except:
        pytest.fail("Can't unlock attribute")
    assert not cd.is_locked("values")
    try:
        cd.lock("values")
    except:
        pytest.fail("Can't lock attribute")


def test_lock_continuousdata_default(draeger1: Sequence):
    try:
        cd = next(iter(draeger1.continuous_data.values()))
    except:
        pytest.fail("No continuous data available in sequence")

    assert cd.is_locked("values")
    assert cd.is_locked("time")

    cd.unlock()
    assert not cd.is_locked("values")
    assert not cd.is_locked("time")

    cd.lock()
    assert cd.is_locked("values")
    assert cd.is_locked("time")


def test_lock_continuousdata_all(draeger1: Sequence):
    try:
        cd = next(iter(draeger1.continuous_data.values()))
    except:
        pytest.fail("No continuous data available in sequence")

    lockables = {attr for attr in vars(cd) if cd.is_lockable(attr)}
    assert lockables == {"values", "time"}

    cd.unlock_all()
    for attr in lockables:
        assert not cd.is_locked(attr)

    cd.lock_all()
    for attr in lockables:
        assert cd.is_locked(attr)


def test_lock_overwrite(draeger1: Sequence) -> None:
    try:
        cd = next(iter(draeger1.continuous_data.values()))
    except:
        pytest.fail("No continuous data available in sequence")

    cd.lock("values")
    with pytest.raises(AttributeError):
        cd.values = "some value"

    cd.unlock("values")
    try:
        old_values = cd.values
        cd.values = "some value"
        cd.values = old_values
    except AttributeError:
        pytest.fail("cd.values should be writable")


def test_lock_not_lockable(draeger1):
    try:
        cd = next(iter(draeger1.continuous_data.values()))
    except:
        pytest.fail("No continuous data available in sequence")

    with pytest.raises(ValueError):
        cd.lock("label")

    with pytest.raises(ValueError):
        cd.lock("unit")

    with pytest.raises(AttributeError):
        cd.lock("non-existing attribute")

    with pytest.raises(ValueError):
        cd.unlock("label")

    with pytest.raises(ValueError):
        cd.unlock("unit")

    with pytest.raises(AttributeError):
        cd.unlock("non-existing attribute")
