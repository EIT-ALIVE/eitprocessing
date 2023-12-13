from dataclasses import dataclass
from eitprocessing.eit_data import EITData
from eitprocessing.eit_data.vendor import Vendor
import pytest


@pytest.fixture
def EITDataSubA():
    @dataclass
    class EITDataSubA(EITData):
        def _from_path(self):
            return self

    return EITDataSubA


def test_init(EITDataSubA):
    with pytest.raises(TypeError):
        # you should not be able to initialize the abstract base class EITData
        _ = EITData()

    with pytest.raises(TypeError):
        # you should not be able to initialize the class without arguments
        _ = EITDataSubA()

    _ = EITDataSubA(
        path="path", nframes=10, time=[1, 2, 3], framerate=20, vendor=Vendor.DRAEGER
    )
