import os
from pathlib import Path

import pytest

from eitprocessing.datahandling.sequence import Sequence

environment = os.environ.get(
    "EIT_PROCESSING_TEST_DATA",
    Path.resolve(Path(__file__).parent.parent),
)
data_directory = Path(environment) / "tests" / "test_data"
draeger_wrapped_time_axis_file = data_directory / "Draeger_wrapped_time_axis.bin"

data_directory = Path(environment) / "test_data"  # overwrite for new style tests
pytest_plugins = [
    "tests.fixtures.eitdata",  # load fixtures from different modules as 'plugins' as workaround
]

dummy_file = data_directory / "not_a_file.dummy"


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


# TODO: Replace request.getfixturevalue() with sequence where possible in other tests
@pytest.fixture
def sequence(request: pytest.FixtureRequest) -> Sequence:
    """Return a Sequence fixture."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def sequence_path(request: pytest.FixtureRequest) -> Path:
    """Return a Sequence fixture."""
    return request.getfixturevalue(request.param)
