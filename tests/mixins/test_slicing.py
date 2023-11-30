import os
import numpy as np
import pytest
from eitprocessing.eit_data.draeger import DraegerEITData


environment = os.environ.get(
    "EIT_PROCESSING_TEST_DATA",
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
)
data_directory = os.path.join(environment, "test_data")
draeger_file1 = os.path.join(data_directory, "Draeger_Test3.bin")
draeger_file2 = os.path.join(data_directory, "Draeger_Test.bin")
timpel_file = os.path.join(data_directory, "Timpel_Test.txt")
dummy_file = os.path.join(data_directory, "not_a_file.dummy")


@pytest.fixture(scope="module")
def draeger_data1():
    return DraegerEITData.from_path(draeger_file1)


def test_slicing(draeger_data1):
    cutoff = 10
    data: DraegerEITData = draeger_data1

    assert data[cutoff] == data[cutoff]
    assert data[0:cutoff] == data[:cutoff]
    # assert data[cutoff : len(data)] == data[cutoff:]

    # assert Sequence.merge(data[:cutoff], data[cutoff:]) == data
    # assert len(data[:cutoff]) == cutoff

    # assert len(data) == len(data[cutoff:]) + len(data[-cutoff:])
    # assert len(data) == len(data[:cutoff]) + len(data[:-cutoff])
