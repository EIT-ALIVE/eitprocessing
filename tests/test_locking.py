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
    assert cd.islockable("values")
    assert cd.islocked("values")
    assert cd.islocked("time")
    try:
        cd.unlock("values")
    except:
        pytest.fail("Can't unlock attribute")
    assert not cd.islocked("values")
    assert cd.islocked("time")
    try:
        cd.lock("values")
    except:
        pytest.fail("Can't lock attribute")
    assert cd.islocked("values")
    assert cd.islocked("time")


def test_lock_continuousdata_default(draeger1: Sequence):
    try:
        cd = next(iter(draeger1.continuous_data.values()))
    except:
        pytest.fail("No continuous data available in sequence")

    assert cd.islocked("values")
    assert cd.islocked("time")

    cd.unlock()
    assert not cd.islocked("values")
    assert not cd.islocked("time")

    cd.lock()
    assert cd.islocked("values")
    assert cd.islocked("time")


def test_lock_continuousdata_all(draeger1: Sequence):
    try:
        cd = next(iter(draeger1.continuous_data.values()))
    except:
        pytest.fail("No continuous data available in sequence")

    lockables = {attr for attr in vars(cd) if cd.islockable(attr)}
    assert lockables == {"values", "time"}

    cd.unlock_all()
    for attr in lockables:
        assert not cd.islocked(attr)

    cd.lock_all()
    for attr in lockables:
        assert cd.islocked(attr)


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
