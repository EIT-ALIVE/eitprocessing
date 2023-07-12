"""Tests for the eitprocessing.my_module module.
"""
import os
import pytest
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
    reading1 = Sequence.from_path(sample_data1, vendor="draeger")
    reading2 = Sequence.from_path(sample_data2, vendor="draeger")
    merged = Sequence.merge(reading1, reading2)
    assert merged.n_frames == (reading1.n_frames + reading2.n_frames)


def test_from_path_1():
    reading = Sequence.from_path(sample_data1, vendor="draeger")
    assert reading.framerate == 20


def test_from_path_2():
    reading = Sequence.from_path(sample_data2, vendor="draeger")
    assert len(reading.time) == 12000
