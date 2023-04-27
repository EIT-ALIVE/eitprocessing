"""Tests for the eitprocessing.my_module module.
"""
import os
import pytest
from eitprocessing.binreader.sequence import Sequence


#import unittest

# sample data needs to be reset to potentially come from 
# a container if we will not share samples
sample_data1 = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'test_data',
    'Testdata_FCVstudy.bin',
)

sample_data2 = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'test_data',
    'Testdata_PEEPtrial.bin',
)


def test_merge():
    reading1 =Sequence.from_path(sample_data1, framerate=20)
    reading2 =Sequence.from_path(sample_data2, framerate=20)
    merged = Sequence.merge(reading1, reading2)
    assert merged.n_frames == (reading1.n_frames +reading2.n_frames)
    

def test__from_path_1():
    reading =Sequence.from_path(sample_data1, framerate=20)
    assert reading.framerate == 20


# def test_hello_with_error():
#     with pytest.raises(ValueError) as excinfo:
#         hello('nobody')
#     assert 'Can not say hello to nobody' in str(excinfo.value)

def test_from_path_2():
    reading =Sequence.from_path(sample_data2, framerate=20)
    assert len(reading.time) == 27680
