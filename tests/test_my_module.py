"""Tests for the eitprocessing.my_module module.
"""
import os
from eitprocessing.binreader.sequence import Sequence


# import unittest

# sample data needs to be reset to potentially come from
# a container if we will not share samples

data_directory = os.environ.get(
    "EIT_PROCESSING_TEST_DATA",
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
)

sample_data1 = os.path.join(
    data_directory,
    "test_data",
    "Testdata.bin",
)

sample_data2 = os.path.join(
    data_directory,
    "test_data",
    "Testdata2.bin",
)


def test_merge():
    # print(sample_data1)
    reading1 = Sequence.from_path(sample_data1, framerate=20, vendor="draeger")
    reading2 = Sequence.from_path(sample_data2, framerate=20, vendor="draeger")
    merged = Sequence.merge(reading1, reading2)
    assert merged.n_frames == (reading1.n_frames + reading2.n_frames)


def test__from_path_1():
    reading = Sequence.from_path(sample_data1, framerate=20, vendor="draeger")
    assert reading.framerate == 20


# def test_hello_with_error():
#     with pytest.raises(ValueError) as excinfo:
#         hello('nobody')
#     assert 'Can not say hello to nobody' in str(excinfo.value)


def test_from_path_2():
    reading = Sequence.from_path(sample_data2, framerate=20, vendor="draeger")
    assert len(reading.time) == 12000
