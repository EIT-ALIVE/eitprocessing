import numpy as np
import pytest
import scipy.signal

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.mixins.arrays import NotAnArray
from eitprocessing.datahandling.pixelmap import PixelMap
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.datahandling.sparsedata import SparseData
from eitprocessing.roi import PixelMask
from eitprocessing.roi.pixelmaskcollection import PixelMaskCollection


def _continuous_data() -> ContinuousData:
    return ContinuousData(
        label="cd",
        name="cd",
        unit="a.u.",
        category="impedance",
        time=np.arange(10.0),
        values=np.arange(10.0),
        sample_frequency=1.0,
    )


def _eit_data() -> EITData:
    return EITData(
        path="somewhere",
        nframes=4,
        time=np.arange(4.0),
        sample_frequency=20.0,
        vendor=Vendor.DRAEGER,
        pixel_impedance=np.ones((4, 2, 2)),
        suppress_simulated_warning=True,
    )


def _sparse_data() -> SparseData:
    return SparseData(label="sd", name="sd", unit=None, category="breath", time=np.arange(3.0), values=[1, 2, 3])


def _interval_data() -> IntervalData:
    return IntervalData(label="id", name="id", unit=None, category="breath", intervals=[(0.0, 1.0)], values=[1])


# Objects that hold their data in a marked field, with the name that should be suggested in the error message.
OBJECTS_WITH_ATTRIBUTE = {
    "PixelMap": (PixelMap([[1.0, 2.0], [3.0, 4.0]]), "values"),
    "PixelMask": (PixelMask(np.ones((2, 2))), "mask"),
    "ContinuousData": (_continuous_data(), "values"),
    "EITData": (_eit_data(), "pixel_impedance"),
    "SparseData": (_sparse_data(), "values"),
    "IntervalData": (_interval_data(), "values"),
}

# Objects without a single data field: the error message should not suggest an attribute.
OBJECTS_WITHOUT_ATTRIBUTE = {
    "Sequence": Sequence(),
    "DataCollection": DataCollection(EITData),
    "PixelMaskCollection": PixelMaskCollection(),
}

ALL_OBJECTS = [*OBJECTS_WITH_ATTRIBUTE.values(), *((obj, None) for obj in OBJECTS_WITHOUT_ATTRIBUTE.values())]


@pytest.mark.parametrize(("obj", "attribute"), OBJECTS_WITH_ATTRIBUTE.values(), ids=OBJECTS_WITH_ATTRIBUTE)
def test_array_attribute_is_found_through_field_metadata(obj: NotAnArray, attribute: str):
    assert obj._array_attribute == attribute


@pytest.mark.parametrize("obj", OBJECTS_WITHOUT_ATTRIBUTE.values(), ids=OBJECTS_WITHOUT_ATTRIBUTE)
def test_array_attribute_is_none_without_marked_field(obj: NotAnArray):
    assert obj._array_attribute is None


@pytest.mark.parametrize(("obj", "attribute"), ALL_OBJECTS, ids=[*OBJECTS_WITH_ATTRIBUTE, *OBJECTS_WITHOUT_ATTRIBUTE])
def test_conversion_to_array_is_refused(obj: NotAnArray, attribute: str | None):
    """`__array__`: `numpy.asarray()` and friends, and therefore most of scipy."""
    for convert in (np.asarray, np.array, lambda o: np.array(o, dtype=float)):
        with pytest.raises(TypeError, match=f"`{type(obj).__name__}` objects can not be used as an array") as excinfo:
            convert(obj)

        if attribute:
            assert f"Pass the `{attribute}` attribute instead." in str(excinfo.value)
        else:
            assert "Pass the" not in str(excinfo.value)


@pytest.mark.parametrize(("obj", "attribute"), ALL_OBJECTS, ids=[*OBJECTS_WITH_ATTRIBUTE, *OBJECTS_WITHOUT_ATTRIBUTE])
def test_ufuncs_are_refused(obj: NotAnArray, attribute: str | None):  # noqa: ARG001
    """`__array_ufunc__`: `numpy.sin()`, `numpy.add()`, `array + object`, ..."""
    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.sin`\)"):
        np.sin(obj)


@pytest.mark.parametrize(
    ("obj", "attribute"),
    # PixelMap forwards binary operators to its own reflected operators, which raise a more specific message.
    [(obj, attribute) for obj, attribute in ALL_OBJECTS if not isinstance(obj, PixelMap)],
    ids=[name for name in [*OBJECTS_WITH_ATTRIBUTE, *OBJECTS_WITHOUT_ATTRIBUTE] if name != "PixelMap"],
)
def test_binary_operators_with_an_array_are_refused(obj: NotAnArray, attribute: str | None):  # noqa: ARG001
    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.add`\)"):
        np.add(np.ones(2), obj)

    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.add`\)"):
        _ = np.ones(2) + obj


@pytest.mark.parametrize(("obj", "attribute"), ALL_OBJECTS, ids=[*OBJECTS_WITH_ATTRIBUTE, *OBJECTS_WITHOUT_ATTRIBUTE])
def test_array_functions_are_refused(obj: NotAnArray, attribute: str | None):  # noqa: ARG001
    """`__array_function__`: the rest of the numpy API, which dispatches before conversion."""
    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.mean`\)"):
        np.mean(obj)

    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.concatenate`\)"):
        np.concatenate([obj, obj])

    with pytest.raises(TypeError, match=r"can not be used as an array \(attempted `numpy.stack`\)"):
        np.stack([obj])


@pytest.mark.parametrize(("obj", "attribute"), ALL_OBJECTS, ids=[*OBJECTS_WITH_ATTRIBUTE, *OBJECTS_WITHOUT_ATTRIBUTE])
def test_scipy_functions_are_refused(obj: NotAnArray, attribute: str | None):  # noqa: ARG001
    """Scipy converts its input to an array before doing anything else, so `__array__` covers it."""
    with pytest.raises(TypeError, match="can not be used as an array"):
        scipy.signal.detrend(obj)

    with pytest.raises(TypeError, match="can not be used as an array"):
        scipy.signal.filtfilt([1.0], [1.0], obj)


def test_slicing_is_unaffected():
    """The mixin should only block numpy; regular Python behaviour must keep working."""
    continuous_data = _continuous_data()

    assert len(continuous_data) == 10
    assert len(continuous_data[2:5]) == 3
    assert np.array_equal(continuous_data[2:5].values, np.arange(2.0, 5.0))
    assert continuous_data == _continuous_data()
    assert continuous_data.t[2.0:5.0] == continuous_data[2:5]


def test_slicing_does_not_recurse():
    """Without the mixin, numpy walks `__getitem__`, copying the object once per index."""
    continuous_data = _continuous_data()

    with pytest.raises(TypeError, match="can not be used as an array"):
        np.asarray(continuous_data)


def test_underlying_data_can_still_be_used():
    """The suggestion in the error message should actually work."""
    continuous_data = _continuous_data()

    assert np.mean(getattr(continuous_data, continuous_data._array_attribute)) == pytest.approx(4.5)
    assert np.asarray(PixelMask(np.ones((2, 2))).mask).shape == (2, 2)
