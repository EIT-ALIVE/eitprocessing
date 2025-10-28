from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence
from tests.conftest import timpel_file

# ruff: noqa: ERA001  #TODO: remove this line


def test_loading_timpel(
    timpel1: Sequence,
    # timpel_double: Sequence,  # does not currently work, because it won't load due to the time axes overlapping
):
    using_vendor = load_eit_data(timpel_file, vendor=Vendor.TIMPEL, label="timpel")
    assert timpel1 == using_vendor
    assert isinstance(timpel1, Sequence)
    assert isinstance(timpel1.eit_data["raw"], EITData)
    assert timpel1.eit_data["raw"].vendor == Vendor.TIMPEL

    # Load multiple
    # assert isinstance(timpel_double, Sequence)
    # assert len(timpel_double) == 2 * len(timpel1)


def test_load_partial(
    timpel1: Sequence,
):
    cutoff = 58

    # Timpel
    timpel_part1 = load_eit_data(timpel_file, vendor="timpel", max_frames=cutoff, label="timpel_part_1")
    timpel_part2 = load_eit_data(timpel_file, vendor="timpel", first_frame=cutoff, label="timpel_part2")

    assert len(timpel_part1) == cutoff
    assert len(timpel_part2) == len(timpel1) - cutoff
    assert timpel_part1 == timpel1[:cutoff]
    assert timpel_part2 == timpel1[cutoff:]
    assert Sequence.concatenate(timpel_part1, timpel_part2) == timpel1
    # assert Sequence.concatenate(timpel_part2, timpel_part1) != timpel1
