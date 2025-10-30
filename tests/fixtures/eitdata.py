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
def sentec_healthy_volunteer_1a_path() -> Path:
    return data_directory / "sentec_healthy_volunteer_1a.zri"


@pytest.fixture
def sentec_healthy_volunteer_1b_path() -> Path:
    return data_directory / "sentec_healthy_volunteer_1b.zri"


@pytest.fixture
def sentec_healthy_volunteer_2a_path() -> Path:
    return data_directory / "sentec_healthy_volunteer_2a.zri"


@pytest.fixture
def sentec_healthy_volunteer_2b_path() -> Path:
    return data_directory / "sentec_healthy_volunteer_2b.zri"


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


@pytest.fixture
def sentec_healthy_volunteer_1a(sentec_healthy_volunteer_1a_path: Path) -> Sequence:
    return load_eit_data(
        sentec_healthy_volunteer_1a_path,
        vendor="sentec",
        label="sentec_healthy_volunteer_1a",
    )


@pytest.fixture
def sentec_healthy_volunteer_1b(sentec_healthy_volunteer_1b_path: Path) -> Sequence:
    return load_eit_data(
        sentec_healthy_volunteer_1b_path,
        vendor="sentec",
        label="sentec_healthy_volunteer_1b",
    )


@pytest.fixture
def sentec_healthy_volunteer_1(
    sentec_healthy_volunteer_1a_path: Path, sentec_healthy_volunteer_1b_path: Path
) -> Sequence:
    return load_eit_data(
        [sentec_healthy_volunteer_1a_path, sentec_healthy_volunteer_1b_path],
        vendor="sentec",
        label="sentec_healthy_volunteer_1",
    )


@pytest.fixture
def sentec_healthy_volunteer_2a(sentec_healthy_volunteer_2a_path: Path) -> Sequence:
    return load_eit_data(
        sentec_healthy_volunteer_2a_path,
        vendor="sentec",
        label="sentec_healthy_volunteer_2a",
    )


@pytest.fixture
def sentec_healthy_volunteer_2b(sentec_healthy_volunteer_2b_path: Path) -> Sequence:
    return load_eit_data(
        sentec_healthy_volunteer_2b_path,
        vendor="sentec",
        label="sentec_healthy_volunteer_2b",
    )


@pytest.fixture
def sentec_healthy_volunteer_2(
    sentec_healthy_volunteer_2a_path: Path, sentec_healthy_volunteer_2b_path: Path
) -> Sequence:
    return load_eit_data(
        [sentec_healthy_volunteer_2a_path, sentec_healthy_volunteer_2b_path],
        vendor="sentec",
        label="sentec_healthy_volunteer_2",
    )
