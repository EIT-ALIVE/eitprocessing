from pathlib import Path

import pytest

from eitprocessing.datahandling.loading import load_eit_data


def test_loading_illegal_path():
    for vendor in ["draeger", "timpel"]:
        with pytest.raises(FileNotFoundError):
            _ = load_eit_data("non-existing-path", vendor=vendor, sample_frequency=20)


# TODO: add timpel/sentec data
def test_loading_illegal_vendor(draeger_20hz_healthy_volunteer_path: Path):
    with pytest.raises(OSError):
        # wrong vendor for the file
        _ = load_eit_data(draeger_20hz_healthy_volunteer_path, vendor="timpel")

    with pytest.raises(NotImplementedError):
        _ = load_eit_data(draeger_20hz_healthy_volunteer_path, vendor="non-existing vendor")
