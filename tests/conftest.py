import os
from pathlib import Path

import pytest

from eitprocessing.datahandling.loading import load_eit_data

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


@pytest.fixture(scope="session")
def draeger1():
    return load_eit_data(draeger_file1, vendor="draeger", sample_frequency=20, label="draeger1")


@pytest.fixture(scope="session")
def draeger2():
    return load_eit_data(draeger_file2, vendor="draeger", sample_frequency=20, label="draeger2")


@pytest.fixture(scope="session")
def draeger_both():
    return load_eit_data([draeger_file2, draeger_file1], vendor="draeger", sample_frequency=20, label="draeger_both")


@pytest.fixture(scope="session")
def draeger_pp():
    return load_eit_data(draeger_file_pp, vendor="draeger", sample_frequency=50, label="draeger2")


@pytest.fixture(scope="session")
def timpel1():
    return load_eit_data(timpel_file, vendor="timpel", label="timpel")


@pytest.fixture(scope="session")
def draeger_wrapped_time_axis():
    return load_eit_data(
        draeger_wrapped_time_axis_file, vendor="draeger", sample_frequency=20, label="draeger_wrapped_time_axis"
    )


# @pytest.fixture(scope="session")
# def timpel_double():
#     return load_eit_data([timpel_file, timpel_file], vendor="timpel", label="timpel_double")
