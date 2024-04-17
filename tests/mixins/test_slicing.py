from copy import deepcopy

import pytest

from eitprocessing.datahandling.eitdata import Vendor
from eitprocessing.datahandling.sequence import Sequence

# ruff: noqa: ERA001  #TODO: remove this line


def test_slicing(draeger1: Sequence, timpel1: Sequence):
    cutoff = 100

    for seq in [draeger1, timpel1]:
        assert seq[0:cutoff] == seq[:cutoff]
        assert seq[cutoff : len(seq)] == seq[cutoff:]

        assert len(seq[:cutoff]) == cutoff
        assert len(seq) == len(seq[cutoff:]) + len(seq[-cutoff:])
        assert len(seq) == len(seq[:cutoff]) + len(seq[:-cutoff])

        # concatenated = Sequence.concatenate(seq[:cutoff], seq[cutoff:])
        # concatenated.eit_data["raw"].path = seq.eit_data["raw"].path  # what's this doing??
        # assert concatenated == seq


def test_select_by_time(draeger2: Sequence):
    pytest.skip("selecting by time not finalized yet")

    # test illegal
    with pytest.warns(UserWarning):
        _ = draeger2.select_by_time()
    with pytest.warns(UserWarning):
        _ = draeger2.select_by_time(None, None)
    with pytest.warns(UserWarning):
        _ = draeger2.select_by_time(None)
    with pytest.warns(UserWarning):
        _ = draeger2.select_by_time(end_time=None)

    # TODO (#82): this function is kinda ugly. Would be nice to refactor it
    # but I am struggling to think of a logical way to loop through.
    ms = 1 / 1000

    # test start_time only
    full_length = len(draeger2)
    start_slices = [
        # (slice time, expected missing slices if inclusive=True, expected missing slices if inclusive=False)
        (draeger2.time[22], 22, 22),
        (draeger2.time[22] - ms, 21, 22),
        (draeger2.time[22] + ms, 22, 23),
    ]
    for test_settings in start_slices:
        print(test_settings)  # noqa: T201
        sliced_inc = draeger2.select_by_time(start_time=test_settings[0], start_inclusive=True)
        assert len(sliced_inc) == full_length - test_settings[1]
        sliced_exc = draeger2.select_by_time(start_time=test_settings[0], start_inclusive=False)
        assert len(sliced_exc) == len(draeger2) - test_settings[2]
        # test default:
        assert draeger2.select_by_time(start_time=test_settings[0]) == sliced_inc

    # test end_time only
    end_slices = [
        # (slice time, expected length if inclusive=True, expected length if inclusive=False)
        (draeger2.time[52], 52, 52),
        (draeger2.time[52] - ms, 51, 52),
        (draeger2.time[52] + ms, 52, 53),
    ]
    for test_settings in end_slices:
        print(test_settings)  # noqa: T201
        sliced_inc = draeger2.select_by_time(end_time=test_settings[0], end_inclusive=True)
        assert len(sliced_inc) == test_settings[1]
        sliced_exc = draeger2.select_by_time(end_time=test_settings[0], end_inclusive=False)
        assert len(sliced_exc) == test_settings[2]
        # test default:
        assert draeger2.select_by_time(end_time=test_settings[0]) == sliced_exc

    # test start_time and end_time
    for start_slicing in start_slices:
        for end_slicing in end_slices:
            # True/True
            sliced = draeger2.select_by_time(
                start_time=start_slicing[0],
                end_time=end_slicing[0],
                start_inclusive=True,
                end_inclusive=True,
            )
            assert len(sliced) == end_slicing[1] - start_slicing[1]

            # False/True
            sliced = draeger2.select_by_time(
                start_time=start_slicing[0],
                end_time=end_slicing[0],
                start_inclusive=False,
                end_inclusive=True,
            )
            assert len(sliced) == end_slicing[1] - start_slicing[2]

            # True/False
            sliced = draeger2.select_by_time(
                start_time=start_slicing[0],
                end_time=end_slicing[0],
                start_inclusive=True,
                end_inclusive=False,
            )
            assert len(sliced) == end_slicing[2] - start_slicing[1]

            # False/False
            sliced = draeger2.select_by_time(
                start_time=start_slicing[0],
                end_time=end_slicing[0],
                start_inclusive=False,
                end_inclusive=False,
            )
            assert len(sliced) == end_slicing[2] - start_slicing[2]


def test_concatenate(
    draeger1: Sequence,
    draeger2: Sequence,
    draeger_both: Sequence,
    timpel1: Sequence,
    # timpel_double: Sequence,
):
    merged_draeger = Sequence.concatenate(draeger2, draeger1)
    assert len(merged_draeger.eit_data["raw"]) == len(draeger2.eit_data["raw"]) + len(
        draeger1.eit_data["raw"],
    )
    assert merged_draeger == draeger_both
    added_draeger = draeger2 + draeger1
    assert added_draeger == merged_draeger

    # slice and concatenate
    pytest.skip("slice and concatenate doesn't work")
    cutoff_pont = 100
    part1 = timpel1[:cutoff_pont]
    part2 = timpel1[cutoff_pont:]
    assert timpel1 == Sequence.concatenate(part1, part2)

    # TODO: add tests for:
    # - concatenating a third Sequence on top (or two double-sequences), also checking that path attribute is flat list
    # - as above, but for timpel and sentec


def test_illegal_concatenation(timpel1: Sequence, draeger1: Sequence, draeger2: Sequence):
    # Concatenate wrong order
    _ = Sequence.concatenate(draeger2, draeger1)
    with pytest.raises(ValueError):
        _ = Sequence.concatenate(draeger1, draeger2)

    # Concatenate different vendors
    with pytest.raises(TypeError):
        _ = Sequence.concatenate(timpel1, draeger1)

    # Concatenate different framerate (for EITData)
    draeger1_framerate = deepcopy(draeger1)
    _ = Sequence.concatenate(draeger2, draeger1_framerate)
    draeger1_framerate.eit_data["raw"].framerate = 50
    with pytest.raises(ValueError):
        _ = Sequence.concatenate(draeger2, draeger1_framerate)

    # Not sure what this one is testing exactly.
    # My guess is that adjusting the vendor of an EIData instance should not be allowed once it has been instantiated.
    draeger1_vendor = deepcopy(draeger1)
    draeger1_vendor.eit_data["raw"].vendor = Vendor.TIMPEL
    with pytest.raises(ValueError):
        # TODO (#77): update this to AttributeError, once equivalence check for framesets is implemented.
        _ = Sequence.concatenate(draeger1, timpel1)
