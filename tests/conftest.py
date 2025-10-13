import os
from pathlib import Path

import pytest

from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence

# ruff: noqa: ERA001  #TODO: remove this line

environment = os.environ.get(
    "EIT_PROCESSING_TEST_DATA",
    Path.resolve(Path(__file__).parent.parent),
)
data_directory = Path(environment) / "tests" / "test_data"
draeger_file1 = data_directory / "Draeger_Test3.bin"
draeger_file2 = data_directory / "Draeger_Test.bin"
draeger_file3 = data_directory / "Draeger_Test_event_on_first_frame.bin"
draeger_wrapped_time_axis_file = data_directory / "Draeger_wrapped_time_axis.bin"
draeger_file_pp = data_directory / "Draeger_PP_data.bin"
timpel_file = data_directory / "Timpel_test.txt"
dummy_file = data_directory / "not_a_file.dummy"

data_directory = Path(environment) / "testdata"  # overwrite for new style tests
pytest_plugins = [
    "tests.fixtures.eitdata",  # load fixtures from different modules as 'plugins' as workaround
]


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run tests marked as slow")


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def draeger1() -> Sequence:
    return load_eit_data(draeger_file1, vendor="draeger", sample_frequency=20, label="draeger1")


@pytest.fixture(scope="session")
def draeger2() -> Sequence:
    return load_eit_data(draeger_file2, vendor="draeger", sample_frequency=20, label="draeger2")


@pytest.fixture(scope="session")
def draeger_both() -> Sequence:
    return load_eit_data([draeger_file2, draeger_file1], vendor="draeger", sample_frequency=20, label="draeger_both")


@pytest.fixture(scope="session")
def draeger_pp() -> Sequence:
    return load_eit_data(draeger_file_pp, vendor="draeger", sample_frequency=50, label="draeger2")


@pytest.fixture(scope="session")
def timpel1() -> Sequence:
    return load_eit_data(timpel_file, vendor="timpel", label="timpel")


@pytest.fixture(scope="session")
def draeger_wrapped_time_axis() -> Sequence:
    return load_eit_data(
        draeger_wrapped_time_axis_file, vendor="draeger", sample_frequency=20, label="draeger_wrapped_time_axis"
    )


# @pytest.fixture(scope="session")
# def timpel_double():
#     return load_eit_data([timpel_file, timpel_file], vendor="timpel", label="timpel_double")


# TODO: Replace request.getfixturevalue() with sequence where possible in other tests
@pytest.fixture
def sequence(request: pytest.FixtureRequest) -> Sequence:
    """Return a Sequence fixture."""
    return request.getfixturevalue(request.param)
