import contextlib

import numpy as np
import pytest

from eitprocessing.datahandling.eitdata import EITData


@pytest.fixture
def frozen_eitdata_object() -> EITData:
    return EITData(
        label="test_label",
        time=np.arange(10) / 10.0,
        values=np.random.default_rng().random((10, 3, 3)),
        sample_frequency=10.0,
        vendor="simulated",
    )


def test_frozen_time_axis(frozen_eitdata_object: EITData):
    with pytest.raises(AttributeError, match="cannot assign to field 'vendor'"):
        frozen_eitdata_object.vendor = "new_vendor"

    with pytest.raises(ValueError, match="output array is read-only"):
        frozen_eitdata_object.time += 1.0

    with pytest.raises(AttributeError, match="cannot assign to field 'time'"):
        frozen_eitdata_object.time = frozen_eitdata_object.time + 1.0

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        frozen_eitdata_object.time[0] = 1.0


def test_frozen_values(frozen_eitdata_object: EITData):
    with pytest.raises(ValueError, match="output array is read-only"):
        frozen_eitdata_object.values += 1.0

    with pytest.raises(AttributeError, match="cannot assign to field 'values'"):
        frozen_eitdata_object.values = frozen_eitdata_object.values + 1.0

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        frozen_eitdata_object.values[0, 0, 0] = 1.0


def test_unfreeze_array_on_copy(frozen_eitdata_object: EITData):
    values_copy = frozen_eitdata_object.values.copy()
    assert values_copy.flags["WRITEABLE"]
    values_copy += 1.0
    new_frozen_eitdata_object = frozen_eitdata_object.update(values=values_copy)
    assert not new_frozen_eitdata_object.values.flags["WRITEABLE"]
    assert np.array_equal(values_copy, new_frozen_eitdata_object.values)


def test_frozen_slice(frozen_eitdata_object: EITData):
    values_view = frozen_eitdata_object.values[:5, :, :]
    assert not values_view.flags["WRITEABLE"]
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        values_view[0, 0, 0] = 1.0

    values_view = values_view.copy()
    assert values_view.flags["WRITEABLE"]
    values_view += 1.0


def test_cannot_unfreeze(frozen_eitdata_object: EITData):
    base = frozen_eitdata_object.values.base
    with contextlib.suppress(AttributeError):
        while True:
            base = base.base

    if not isinstance(base, memoryview):
        pytest.skip("Array is not based on a memoryview; cannot test unfreeze.")

    with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True of this array"):
        frozen_eitdata_object.values.flags["WRITEABLE"] = True
