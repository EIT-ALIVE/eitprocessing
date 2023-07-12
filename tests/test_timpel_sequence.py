import copy
import os
from eitprocessing.binreader.sequence import Sequence
from eitprocessing.binreader.sequence import TimpelSequence
from eitprocessing.binreader.sequence import Vendor


data_directory = os.environ.get(
    "EIT_PROCESSING_TEST_DATA",
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
)

print(__file__)

sample_data1 = os.path.join(data_directory, "test_data", "testdata_timpel.txt")


def test_from_path():
    using_vendor = Sequence.from_path(sample_data1, vendor=Vendor.TIMPEL)
    using_str = Sequence.from_path(sample_data1, vendor="timpel")

    assert using_vendor == using_str
    assert isinstance(using_str, TimpelSequence)
    assert isinstance(using_vendor, TimpelSequence)


def test_equals():
    full_data = Sequence.from_path(sample_data1, vendor="timpel")

    full_data_copy = TimpelSequence()
    full_data_copy.path = copy.deepcopy(full_data.path)
    full_data_copy.time = copy.deepcopy(full_data.time)
    full_data_copy.nframes = copy.deepcopy(full_data.nframes)
    full_data_copy.framerate = copy.deepcopy(full_data.framerate)
    full_data_copy.framesets = copy.deepcopy(full_data.framesets)
    full_data_copy.events = copy.deepcopy(full_data.events)
    full_data_copy.timing_errors = copy.deepcopy(full_data.timing_errors)
    full_data_copy.phases = copy.deepcopy(full_data.phases)
    full_data_copy.vendor = copy.deepcopy(full_data.vendor)

    assert full_data_copy == full_data

    # test whether a difference in phases fails equality test
    full_data_copy.phases.append(full_data_copy.phases[-1])
    assert full_data != full_data_copy
    full_data_copy.phases = copy.deepcopy(full_data.phases)

    full_data_copy.phases[0].index += 1
    assert full_data != full_data_copy
    full_data_copy.phases = copy.deepcopy(full_data.phases)

    # test wheter a difference in framesets fails equality test
    full_data_copy.framesets["test"] = full_data_copy.framesets["raw"].deepcopy()
    assert full_data != full_data_copy
    full_data_copy.framesets = copy.deepcopy(full_data.framesets)

    full_data_copy.framesets["raw"].name += "_"
    assert full_data != full_data_copy
    full_data_copy.framesets = copy.deepcopy(full_data.framesets)


def test_copy():
    full_data = Sequence.from_path(sample_data1, vendor="timpel")
    full_data_copy = full_data.deepcopy()

    assert full_data == full_data_copy


def test_slicing():
    full_data = Sequence.from_path(sample_data1, vendor="timpel")

    assert full_data[:100] == full_data[:100]  # tests whether slicing alters full_data
    assert full_data[0:100] == full_data[:100]
    assert full_data[100 : len(full_data)] == full_data[100:]


def test_limit_frames_merged_equals_full_data():
    full_data = Sequence.from_path(sample_data1, vendor="timpel")
    limit_first_part = Sequence.from_path(sample_data1, nframes=100, vendor="timpel")
    limit_second_part = Sequence.from_path(sample_data1, first_frame=100, vendor="timpel")

    assert limit_first_part == full_data[:100]
    assert limit_second_part == full_data[100:]
    assert Sequence.merge(limit_first_part, limit_second_part) == full_data
