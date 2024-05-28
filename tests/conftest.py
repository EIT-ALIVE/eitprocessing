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
draeger_file1 = Path(data_directory) / "Draeger_Test3.bin"
draeger_file2 = Path(data_directory) / "Draeger_Test.bin"
timpel_file = Path(data_directory) / "Timpel_test.txt"
dummy_file = Path(data_directory) / "not_a_file.dummy"


@pytest.fixture(scope="session")
def draeger1():
    return load_eit_data(draeger_file1, vendor="draeger", label="draeger1")


@pytest.fixture(scope="session")
def draeger2():
    return load_eit_data(draeger_file2, vendor="draeger", label="draeger2")


@pytest.fixture(scope="session")
def draeger_both():
    return load_eit_data([draeger_file2, draeger_file1], vendor="draeger", label="draeger_both")


@pytest.fixture(scope="session")
def timpel1():
    return load_eit_data(timpel_file, vendor="timpel", label="timpel")


# @pytest.fixture(scope="session")
# def timpel_double():
#     return load_eit_data([timpel_file, timpel_file], vendor="timpel", label="timpel_double")
