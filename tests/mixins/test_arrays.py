import numpy as np
import pytest
import scipy.signal

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.mixins.arrays import NotAnArray


@pytest.mark.parametrize(
    ("data_object", "attribute"),
    [
        ("pixel_map", "values"),
        ("pixel_mask", "mask"),
        ("continuous_data", "values"),
        ("eit_data", "pixel_impedance"),
        ("sparse_data", "values"),
        ("interval_data", "values"),
        ("empty_sequence", None),
        ("data_collection", None),
        ("pixel_mask_collection", None),
    ],
    indirect=["data_object"],
)
def test_error_names_the_data_attribute(data_object: NotAnArray, attribute: str | None):
    """Every object using the mixin refuses conversion and points at its own data attribute, if it has one."""
    with pytest.raises(
        TypeError, match=f"`{type(data_object).__name__}` objects can not be used as an array"
    ) as excinfo:
        np.asarray(data_object)

    if attribute:
        assert f"Pass the `{attribute}` attribute instead." in str(excinfo.value)
    else:
        assert "Pass the" not in str(excinfo.value)


def test_conversion_to_array_is_refused(continuous_data: ContinuousData):
    """`__array__`: `numpy.asarray()` and friends."""
    for convert in (np.asarray, np.array, lambda obj: np.array(obj, dtype=float)):
        with pytest.raises(TypeError, match="can not be used as an array"):
            convert(continuous_data)


def test_ufuncs_are_refused(continuous_data: ContinuousData):
    """`__array_ufunc__`: `numpy.sin()`, `numpy.add()`, `array + object`, ..."""
    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.sin`\)"):
        np.sin(continuous_data)

    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.add`\)"):
        np.add(np.ones(10), continuous_data)

    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.add`\)"):
        _ = np.ones(10) + continuous_data


def test_array_functions_are_refused(continuous_data: ContinuousData):
    """`__array_function__`: the rest of the numpy API, which dispatches before conversion."""
    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.mean`\)"):
        np.mean(continuous_data)

    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.concatenate`\)"):
        np.concatenate([continuous_data, continuous_data])

    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.stack`\)"):
        np.stack([continuous_data])


def test_scipy_functions_are_refused(continuous_data: ContinuousData):
    """Scipy converts its input to an array before doing anything else, so `__array__` covers it."""
    with pytest.raises(TypeError, match="can not be used as an array"):
        scipy.signal.detrend(continuous_data)

    with pytest.raises(TypeError, match="can not be used as an array"):
        scipy.signal.filtfilt([1.0], [1.0], continuous_data)


def test_slicing_is_unaffected(continuous_data: ContinuousData):
    """The mixin should only block numpy; regular Python behaviour must keep working."""
    assert len(continuous_data) == 10
    assert len(continuous_data[2:5]) == 3
    assert np.array_equal(continuous_data[2:5].values, np.arange(2.0, 5.0))
    assert continuous_data.t[2.0:5.0] == continuous_data[2:5]


def test_underlying_data_can_still_be_used(continuous_data: ContinuousData):
    """The suggestion in the error message should actually work."""
    assert np.mean(continuous_data.values) == pytest.approx(4.5)
