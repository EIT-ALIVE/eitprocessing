import pytest

from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence
from tests.conftest import draeger_file1, draeger_file2, dummy_file, timpel_file

# ruff: noqa: ERA001


def test_loading_draeger(
    draeger_data1: Sequence,
    draeger_data2: Sequence,
    draeger_data_both: Sequence,
):
    assert isinstance(draeger_data1, Sequence)
    assert isinstance(draeger_data1.eit_data["raw"], EITData)
    assert draeger_data1.eit_data["raw"].framerate == 20
    assert len(draeger_data1.eit_data["raw"]) == len(draeger_data1.eit_data["raw"].time)
    assert len(draeger_data2.eit_data["raw"].time) == 20740

    assert draeger_data1 == load_eit_data(draeger_file1, vendor="draeger", label="draeger1")
    assert draeger_data1 != load_eit_data(draeger_file1, vendor="draeger", label="something_else")
    assert draeger_data1 != draeger_data2

    # Load multiple
    assert len(draeger_data_both.eit_data["raw"]) == len(draeger_data1.eit_data["raw"]) + len(
        draeger_data2.eit_data["raw"],
    )

    # draeger_inverted = load_eit_data([draeger_file1, draeger_file2], vendor="draeger", label="inverted")
    # assert len(draeger_data_both) == len(draeger_inverted)
    # assert draeger_data_both != draeger_inverted


def test_loading_timpel(
    draeger_data1: Sequence,
    timpel_data: Sequence,
    # timpel_data_double: Sequence,  # does not currently work, because it won't load due to the time axes overlapping
):
    using_vendor = load_eit_data(timpel_file, vendor=Vendor.TIMPEL, label="timpel")
    assert timpel_data == using_vendor
    assert isinstance(timpel_data, Sequence)
    assert isinstance(timpel_data.eit_data["raw"], EITData)
    assert timpel_data.eit_data["raw"].vendor != draeger_data1.eit_data["raw"].vendor

    # Load multiple
    # assert isinstance(timpel_data_double, Sequence)
    # assert len(timpel_data_double) == 2 * len(timpel_data)


def test_loading_illegal():
    # non existing
    for vendor in ["draeger", "timpel"]:
        with pytest.raises(FileNotFoundError):
            _ = load_eit_data(dummy_file, vendor=vendor)

    # incorrect vendor
    with pytest.raises(OSError):
        _ = load_eit_data(draeger_file1, vendor="timpel")
    with pytest.raises(OSError):
        _ = load_eit_data(timpel_file, vendor="draeger")


def test_load_partial(
    draeger_data2: Sequence,
    timpel_data: Sequence,
):
    cutoff = 58
    # Keep cutoff at 58 for draeger_data2 as there is an event mark at this
    # timepoint. Starting the load specifically at the timepoint of an event
    # marker was tricky to implement, so keeping this cutoff will ensure that
    # code keeps working for this fringe situation.

    # TODO (#81): test what happens if a file has an actual event marker on the very
    # first frame. I suspect this will not be loaded, but I don't have a test
    # file for this situation.

    # Timpel
    timpel_first_part = load_eit_data(timpel_file, "timpel", max_frames=cutoff, label="part_1")
    timpel_second_part = load_eit_data(timpel_file, "timpel", first_frame=cutoff, label="part2")

    assert timpel_first_part == timpel_data[:cutoff]
    assert timpel_second_part == timpel_data[cutoff:]
    assert Sequence.merge(timpel_first_part, timpel_second_part) == timpel_data
    assert Sequence.merge(timpel_second_part, timpel_first_part) != timpel_data

    # Draeger
    draeger_first_part = load_eit_data(draeger_file2, "draeger", max_frames=cutoff)
    draeger_second_part = load_eit_data(draeger_file2, "draeger", first_frame=cutoff)

    assert draeger_first_part == draeger_data2[:cutoff]
    assert draeger_second_part == draeger_data2[cutoff:]
    assert Sequence.merge(draeger_first_part, draeger_second_part) == draeger_data2
    assert Sequence.merge(draeger_second_part, draeger_first_part) != draeger_data2


def test_illegal_first_frame():
    for ff in [0.5, -1, "fdw"]:
        with pytest.raises((TypeError, ValueError)):
            _ = load_eit_data(draeger_file1, "draeger", first_frame=ff)

    for ff2 in [0, 0.0, 1.0, None]:
        _ = load_eit_data(draeger_file1, "draeger", first_frame=ff2)
