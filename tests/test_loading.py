import pytest

from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence
from tests.conftest import draeger_file1, draeger_file2, dummy_file, timpel_file

# ruff: noqa: ERA001  #TODO: remove this line


def test_loading_draeger(
    draeger1: Sequence,
    draeger2: Sequence,
    draeger_both: Sequence,
):
    assert isinstance(draeger1, Sequence)
    assert isinstance(draeger1.eit_data["raw"], EITData)
    assert draeger1.eit_data["raw"].framerate == 20
    assert len(draeger1.eit_data["raw"]) == len(draeger1.eit_data["raw"].time)
    assert len(draeger2.eit_data["raw"].time) == 20740

    assert draeger1 == load_eit_data(draeger_file1, vendor="draeger", label="draeger1")
    assert draeger1 == load_eit_data(draeger_file1, vendor="draeger", label="something_else")
    assert draeger1 != draeger2

    # Load multiple
    assert len(draeger_both.eit_data["raw"]) == len(draeger1.eit_data["raw"]) + len(
        draeger2.eit_data["raw"],
    )

    # test below not possible due to requirement of axis 1 ending before axis b starts
    # draeger_inverted = load_eit_data([draeger_file1, draeger_file2], vendor="draeger", label="inverted")
    # assert len(draeger_both) == len(draeger_inverted)
    # assert draeger_both != draeger_inverted


def test_loading_timpel(
    draeger1: Sequence,
    timpel1: Sequence,
    # timpel_double: Sequence,  # does not currently work, because it won't load due to the time axes overlapping
):
    using_vendor = load_eit_data(timpel_file, vendor=Vendor.TIMPEL, label="timpel")
    assert timpel1 == using_vendor
    assert isinstance(timpel1, Sequence)
    assert isinstance(timpel1.eit_data["raw"], EITData)
    assert timpel1.eit_data["raw"].vendor != draeger1.eit_data["raw"].vendor

    # Load multiple
    # assert isinstance(timpel_double, Sequence)
    # assert len(timpel_double) == 2 * len(timpel1)


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
    draeger2: Sequence,
    timpel1: Sequence,
):
    cutoff = 58
    # Keep cutoff at 58 for draeger2 as there is an event mark at this
    # timepoint. Starting the load specifically at the timepoint of an event
    # marker was tricky to implement, so keeping this cutoff will ensure that
    # code keeps working for this fringe situation.

    # TODO (#81): test what happens if a file has an actual event marker on the very
    # first frame. I suspect this will not be loaded, but I don't have a test
    # file for this situation.

    # Timpel
    timpel_part1 = load_eit_data(timpel_file, "timpel", max_frames=cutoff, label="timpel_part_1")
    timpel_part2 = load_eit_data(timpel_file, "timpel", first_frame=cutoff, label="timpel_part2")

    assert len(timpel_part1) == cutoff
    assert len(timpel_part2) == len(timpel1) - cutoff
    assert timpel_part1 == timpel1[:cutoff]
    assert timpel_part2 == timpel1[cutoff:]
    assert Sequence.concatenate(timpel_part1, timpel_part2) == timpel1
    # assert Sequence.concatenate(timpel_part2, timpel_part1) != timpel1

    # Draeger
    draeger2_part1 = load_eit_data(draeger_file2, "draeger", max_frames=cutoff, label="draeger_part_1")
    draeger2_part2 = load_eit_data(draeger_file2, "draeger", first_frame=cutoff, label="draeger_part_2")

    assert len(draeger2_part1) == cutoff
    assert len(draeger2_part2) == len(draeger2) - cutoff
    assert draeger2_part1 == draeger2[:cutoff]
    pytest.skip(
        "Tests below rely on proper functioning of select_by_time, "
        "which should be refactored before fixing these tests",
    )
    assert draeger2_part2 == draeger2[cutoff:]
    assert Sequence.concatenate(draeger2_part1, draeger2_part2) == draeger2
    # assert Sequence.concatenate(draeger2_part2, draeger2_part1) != draeger2


def test_illegal_first_frame():
    for ff in [0.5, -1, "fdw"]:
        with pytest.raises((TypeError, ValueError)):
            _ = load_eit_data(draeger_file1, "draeger", first_frame=ff)

    for ff2 in [0, 0.0, 1.0, None]:
        _ = load_eit_data(draeger_file1, "draeger", first_frame=ff2)
