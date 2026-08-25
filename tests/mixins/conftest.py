import numpy as np
import pytest

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.pixelmap import PixelMap
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.datahandling.sparsedata import SparseData
from eitprocessing.roi import PixelMask
from eitprocessing.roi.pixelmaskcollection import PixelMaskCollection


@pytest.fixture
def data_object(request: pytest.FixtureRequest) -> object:
    """Return the object named by the parameter, for `indirect` parametrization over several types of object."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def continuous_data() -> ContinuousData:
    """Return a ContinuousData fixture with a short ramp as values."""
    return ContinuousData(
        label="cd",
        name="cd",
        unit="a.u.",
        category="impedance",
        time=np.arange(10.0),
        values=np.arange(10.0),
        sample_frequency=1.0,
    )


@pytest.fixture
def eit_data() -> EITData:
    """Return an EITData fixture with four frames of 2x2 pixels."""
    return EITData(
        path="somewhere",
        nframes=4,
        time=np.arange(4.0),
        sample_frequency=20.0,
        vendor=Vendor.DRAEGER,
        pixel_impedance=np.ones((4, 2, 2)),
        suppress_simulated_warning=True,
    )


@pytest.fixture
def sparse_data() -> SparseData:
    """Return a SparseData fixture with three values."""
    return SparseData(label="sd", name="sd", unit=None, category="breath", time=np.arange(3.0), values=[1, 2, 3])


@pytest.fixture
def interval_data() -> IntervalData:
    """Return an IntervalData fixture with a single interval."""
    return IntervalData(label="id", name="id", unit=None, category="breath", intervals=[(0.0, 1.0)], values=[1])


@pytest.fixture
def empty_sequence() -> Sequence:
    """Return an empty Sequence.

    Named `empty_sequence` to avoid shadowing the `sequence` fixture in `tests/conftest.py`, which loads test data.
    """
    return Sequence()


@pytest.fixture
def data_collection() -> DataCollection:
    """Return an empty DataCollection."""
    return DataCollection(EITData)


@pytest.fixture
def pixel_map() -> PixelMap:
    """Return a PixelMap fixture with 2x2 pixels."""
    return PixelMap([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def pixel_mask() -> PixelMask:
    """Return a PixelMask fixture with 2x2 pixels."""
    return PixelMask(np.ones((2, 2)))


@pytest.fixture
def pixel_mask_collection() -> PixelMaskCollection:
    """Return an empty PixelMaskCollection."""
    return PixelMaskCollection()
