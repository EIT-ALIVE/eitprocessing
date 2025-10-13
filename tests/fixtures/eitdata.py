from pathlib import Path

import pytest

from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence
from tests.conftest import data_directory


@pytest.fixture
def draeger_porcine_1_path() -> Path:
    return data_directory / "draeger_porcine_1.bin"


@pytest.fixture
def draeger_porcine_2_path() -> Path:
    return data_directory / "draeger_porcine_2.bin"


@pytest.fixture
def draeger_porcine_1(draeger_porcine_1_path: Path) -> Sequence:
    return load_eit_data(draeger_porcine_1_path, vendor="draeger", sample_frequency=20, label="draeger_porcine_1")


@pytest.fixture
def draeger_porcine_2(draeger_porcine_2_path: Path) -> Sequence:
    return load_eit_data(draeger_porcine_2_path, vendor="draeger", sample_frequency=20, label="draeger_porcine_2")


@pytest.fixture
def draeger_porcine_1_and_2(draeger_porcine_1_path: Path, draeger_porcine_2_path: Path) -> Sequence:
    return load_eit_data(
        [draeger_porcine_1_path, draeger_porcine_2_path],
        vendor="draeger",
        sample_frequency=20,
        label="draeger_porcine_1_and_2",
    )
