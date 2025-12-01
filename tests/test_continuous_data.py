import dataclasses
import warnings

import numpy as np
import pytest

from eitprocessing.datahandling.continuousdata import ContinuousData


@pytest.fixture
def continuous_data_object():
    n = 100
    sample_frequency = 10
    time = np.arange(n) / sample_frequency
    values = np.arange(n)

    return ContinuousData(
        time=time,
        values=values,
        sample_frequency=sample_frequency,
    )


def test_continuous_data_frozen(continuous_data_object: ContinuousData):
    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field"):
        continuous_data_object.sample_frequency = 20

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        continuous_data_object.time[0] = -1

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        continuous_data_object.values[0] = -1


def test_continuous_data_copy_array(continuous_data_object: ContinuousData):
    time = continuous_data_object.time

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        time[0] = -1

    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        time.flags["WRITEABLE"] = True

    new_time = time.copy()
    new_time[0] = -1

    new_continuous_data_object = continuous_data_object.update(time=new_time)
    assert new_continuous_data_object.time[0] == -1


def test_sample_frequency_deprecation_warning():
    n = 100
    sample_frequency = 10
    time = np.arange(n) / sample_frequency
    values = np.arange(n)

    with pytest.warns(DeprecationWarning, match="`sample_frequency` is set to `None`"):
        ContinuousData(
            time=time,
            values=values,
        )

    with pytest.warns(DeprecationWarning, match="`sample_frequency` is set to `None`"):
        ContinuousData(
            time=time,
            values=values,
            sample_frequency=None,
        )

    with warnings.catch_warnings(record=True) as w:
        ContinuousData(
            time=time,
            values=values,
            sample_frequency=sample_frequency,
        )
        assert len(w) == 0, "No warnings should be raised when sample_frequency is provided"
