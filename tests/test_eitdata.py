from dataclasses import dataclass
from eitprocessing.eit_data import EITData
from eitprocessing.eit_data.vendor import Vendor
from pathlib import Path
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


def test_ensure_path_list_single():
    path_string = "some/path/file.txt"
    path_obj = Path(path_string)

    result_string = EITData._ensure_path_list(path_string)
    result_obj = EITData._ensure_path_list(path_obj)

    assert result_string == [path_obj]
    assert result_obj == [path_obj]


def test_ensure_path_list_multiple():
    paths_string = ["path1/file1.txt", "path2/file2.txt", "path3/file3.txt"]
    paths_obj = [Path(p) for p in paths_string]

    result_string = EITData._ensure_path_list(paths_string)
    result_obj = EITData._ensure_path_list(paths_obj)

    assert result_string == paths_obj
    assert result_obj == paths_obj
