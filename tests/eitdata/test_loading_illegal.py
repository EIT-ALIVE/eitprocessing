from pathlib import Path

import pytest

from eitprocessing.datahandling.loading import load_eit_data


def test_loading_illegal_path():
    # non existing
    for vendor in ["draeger", "timpel"]:
        with pytest.raises(FileNotFoundError):
            _ = load_eit_data("non-existing-path", vendor=vendor, sample_frequency=20)


def test_loading_illegal_vendor(draeger_porcine_1_path: Path):
    with pytest.raises(OSError):
        # wrong vendor for the file
        _ = load_eit_data(draeger_porcine_1_path, vendor="timpel")

    with pytest.raises(NotImplementedError):
        _ = load_eit_data(draeger_porcine_1_path, vendor="non-existing vendor")
