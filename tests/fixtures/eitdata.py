from pathlib import Path

import pytest

from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence
from tests.conftest import data_directory


@pytest.fixture
def draeger_20hz_healthy_volunteer_path() -> Path:
    return data_directory / "draeger_20Hz_healthy_volunteer.bin"


@pytest.fixture
def draeger_20hz_healthy_volunteer_fixed_rr_path() -> Path:
    return data_directory / "draeger_20Hz_healthy_volunteer_fixed_rr.bin"


@pytest.fixture
def draeger_20hz_healthy_volunteer_pressure_pod_path() -> Path:
    return data_directory / "draeger_20Hz_healthy_volunteer_pressure_pod.bin"


@pytest.fixture
def draeger_50hz_healthy_volunteer_pressure_pod_path() -> Path:
    return data_directory / "draeger_50Hz_healthy_volunteer_pressure_pod.bin"


@pytest.fixture
def draeger_20hz_healthy_volunteer(draeger_20hz_healthy_volunteer_path: Path) -> Sequence:
    return load_eit_data(
        draeger_20hz_healthy_volunteer_path,
        vendor="draeger",
        sample_frequency=20,
        label="draeger_20hz_healthy_volunteer",
    )


@pytest.fixture
def draeger_20hz_healthy_volunteer_fixed_rr(draeger_20hz_healthy_volunteer_fixed_rr_path: Path) -> Sequence:
    return load_eit_data(
        draeger_20hz_healthy_volunteer_fixed_rr_path,
        vendor="draeger",
        sample_frequency=20,
        label="draeger_20hz_healthy_volunteer_fixed_rr",
    )


@pytest.fixture
def draeger_20hz_healthy_volunteer_pressure_pod(draeger_20hz_healthy_volunteer_pressure_pod_path: Path) -> Sequence:
    return load_eit_data(
        draeger_20hz_healthy_volunteer_pressure_pod_path,
        vendor="draeger",
        sample_frequency=20,
        label="draeger_20hz_healthy_volunteer_pressure_pod",
    )


@pytest.fixture
def draeger_50hz_healthy_volunteer_pressure_pod(draeger_50hz_healthy_volunteer_pressure_pod_path: Path) -> Sequence:
    return load_eit_data(
        draeger_50hz_healthy_volunteer_pressure_pod_path,
        vendor="draeger",
        sample_frequency=50,
        label="draeger_50hz_healthy_volunteer_pressure_pod",
    )


@pytest.fixture
def draeger_20hz_healthy_volunteer_and_fixed_rr(
    draeger_20hz_healthy_volunteer_path: Path, draeger_20hz_healthy_volunteer_fixed_rr_path: Path
) -> Sequence:
    return load_eit_data(
        [draeger_20hz_healthy_volunteer_path, draeger_20hz_healthy_volunteer_fixed_rr_path],
        vendor="draeger",
        sample_frequency=20,
        label="draeger_20hz_healthy_volunteer_and_fixed_rr",
    )
