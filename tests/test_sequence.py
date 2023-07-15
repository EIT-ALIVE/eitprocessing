"""Tests for the eitprocessing.my_module module.
"""
import copy
import os
import pytest
from eitprocessing.binreader.sequence import Sequence
from eitprocessing.binreader.sequence import TimpelSequence, DraegerSequence
from eitprocessing.binreader.sequence import Vendor

environment = os.environ.get(
    "EIT_PROCESSING_TEST_DATA",
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
)
data_directory = os.path.join(environment, 'test_data')
draeger_file1 = os.path.join(data_directory, "Draeger_Test3.bin")
draeger_file2 = os.path.join(data_directory, "Testdata2.bin")
timpel_file = os.path.join(data_directory, "testdata_timpel.txt")
dummy_file = os.path.join(data_directory, "not_a_file.dummy")


@pytest.fixture(scope='module')
def draeger_data1():
    return Sequence.from_path(draeger_file1, vendor="draeger")

@pytest.fixture(scope='module')
def draeger_data2():
    return Sequence.from_path(draeger_file2, vendor="draeger")

@pytest.fixture(scope='module')
def draeger_data_both():
    return Sequence.from_path([draeger_file1, draeger_file2], vendor="draeger")

@pytest.fixture() #TODO: report bug on pytest -> this fails if scope='module'
def timpel_data():
    return Sequence.from_path(timpel_file, vendor="timpel")

@pytest.fixture()
def timpel_data_double():
    return Sequence.from_path([timpel_file, timpel_file], vendor='timpel')


def test_from_path_draeger(
    draeger_data1: DraegerSequence,
    draeger_data2: DraegerSequence,
    draeger_data_both: DraegerSequence,
    ):
    assert isinstance(draeger_data1, DraegerSequence)
    assert isinstance(draeger_data1, Sequence)
    assert not isinstance (draeger_data1, TimpelSequence)
    assert draeger_data1.framerate == 20
    assert len(draeger_data1) == len(draeger_data1.time)
    assert len(draeger_data2.time) == 12000
    assert draeger_data1 != draeger_data2

    # Load multiple
    assert len(draeger_data_both) == len(draeger_data1) + len(draeger_data2)

    draeger_inverted = Sequence.from_path(
        path=[draeger_file2, draeger_file1],
        vendor='draeger')
    assert len(draeger_data_both) == len(draeger_inverted)
    assert draeger_data_both != draeger_inverted


def test_from_path_timpel(
    draeger_data1: DraegerSequence,
    timpel_data: TimpelSequence,
    timpel_data_double: TimpelSequence,
    ):
    using_vendor = Sequence.from_path(timpel_file, vendor=Vendor.TIMPEL)
    assert timpel_data == using_vendor
    assert isinstance(timpel_data, TimpelSequence)
    assert isinstance(timpel_data, Sequence)
    assert not isinstance(timpel_data, DraegerSequence)
    assert timpel_data.vendor != draeger_data1.vendor

    # Load multiple
    assert isinstance(timpel_data_double, TimpelSequence)
    assert len(timpel_data_double) == 2*len(timpel_data)


def test_illegal_from_path():
    # non existing
    for vendor in ['draeger', 'timpel']:
        with pytest.raises(FileNotFoundError):
            _= Sequence.from_path(dummy_file, vendor=vendor)

    # incorrect vendor
    with pytest.raises(OSError):
        _= Sequence.from_path(draeger_file1, vendor="timpel")
    with pytest.raises(OSError):
        _= Sequence.from_path(timpel_file, vendor="draeger")

    # not implemented
    with pytest.raises(NotImplementedError):
        _= Sequence.from_path(timpel_file, vendor="sentec")


def test_merge(
    draeger_data1: DraegerSequence,
    draeger_data2: DraegerSequence,
    draeger_data_both: DraegerSequence,
    timpel_data: TimpelSequence,
    timpel_data_double: TimpelSequence,
    ):

    merged_draeger = Sequence.merge(draeger_data1, draeger_data2)
    assert len(merged_draeger) == len(draeger_data2) +len(draeger_data1)
    assert merged_draeger == draeger_data_both

    draeger_load_double = Sequence.from_path([draeger_file1, draeger_file1], 'draeger')
    draeger_merge_double = Sequence.merge(draeger_data1, draeger_data1)
    assert draeger_load_double == draeger_merge_double

    merged_timpel = Sequence.merge(timpel_data, timpel_data)
    assert len(merged_timpel) == 2*len(timpel_data)
    assert timpel_data_double == merged_timpel


def test_copy(
    draeger_data1: DraegerSequence,
    timpel_data: TimpelSequence,
    ):
    data: Sequence
    for data in [draeger_data1, timpel_data]:
        print(data.vendor)
        data_copy = data.deepcopy()
        assert data == data_copy


def test_equals(
    draeger_data1: DraegerSequence,
    timpel_data: TimpelSequence,
    ):
    data: Sequence
    for data in [draeger_data1, timpel_data]:
        print(data.vendor)

        data_copy = Sequence()
        data_copy.path = copy.deepcopy(data.path)
        data_copy.time = copy.deepcopy(data.time)
        data_copy.nframes = copy.deepcopy(data.nframes)
        data_copy.framerate = copy.deepcopy(data.framerate)
        data_copy.framesets = copy.deepcopy(data.framesets)
        data_copy.events = copy.deepcopy(data.events)
        data_copy.timing_errors = copy.deepcopy(data.timing_errors)
        data_copy.phases = copy.deepcopy(data.phases)
        data_copy.vendor = copy.deepcopy(data.vendor)

        assert data_copy == data

        # test whether a difference in phases fails equality test
        data_copy.phases.append(data_copy.phases[-1])
        assert data != data_copy
        data_copy.phases = copy.deepcopy(data.phases)

        data_copy.phases[0].index += 1
        assert data != data_copy
        data_copy.phases = copy.deepcopy(data.phases)

        # test wheter a difference in framesets fails equality test
        data_copy.framesets["test"] = data_copy.framesets["raw"].deepcopy()
        assert data != data_copy
        data_copy.framesets = copy.deepcopy(data.framesets)

        data_copy.framesets["raw"].name += "_"
        assert data != data_copy
        data_copy.framesets = copy.deepcopy(data.framesets)


def test_slicing(
    draeger_data1: DraegerSequence,
    timpel_data: TimpelSequence,
    ):
    cutoff = 100

    data: Sequence
    for data in [draeger_data1, timpel_data]:
        print(data.vendor)
        assert data[:cutoff] == data[:cutoff]  # tests whether slicing alters full_data
        assert data[0:cutoff] == data[:cutoff]
        assert data[cutoff : len(data)] == data[cutoff:]

        assert Sequence.merge(data[:cutoff], data[cutoff:]) == data
        assert len(data[:cutoff]) == cutoff


def test_load_partial( #noqa
    draeger_data1: DraegerSequence,
    timpel_data: TimpelSequence,
    ):

    cutoff = 100

    # Timpel
    timpel_first_part = Sequence.from_path(timpel_file, "timpel", nframes=cutoff)
    timpel_second_part = Sequence.from_path(timpel_file, "timpel", first_frame=cutoff)

    assert timpel_first_part == timpel_data[:cutoff]
    assert timpel_second_part == timpel_data[cutoff:]
    assert Sequence.merge(timpel_first_part, timpel_second_part) == timpel_data
    assert Sequence.merge(timpel_second_part, timpel_first_part) != timpel_data

    # Draeger
    # TODO: slicing draeger sequences leads to resetting of phases.time as well
    # as losing events information.
    # This is likely due to the Sequence.select_by_indices method or one of its
    # submethods
    draeger_first_part = Sequence.from_path(draeger_file1, "draeger", nframes=cutoff)
    draeger_second_part = Sequence.from_path(draeger_file1, "draeger", first_frame=cutoff)

    assert draeger_first_part == draeger_data1[:cutoff]
    # assert draeger_second_part == draeger_data1[cutoff:]
    assert Sequence.merge(draeger_first_part, draeger_second_part) == draeger_data1
    assert Sequence.merge(draeger_second_part, draeger_first_part) != draeger_data1
