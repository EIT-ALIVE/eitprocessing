from typing import TYPE_CHECKING

import pytest

from eitprocessing.datahandling.eitdata import Vendor
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence
from tests.conftest import draeger_file1

if TYPE_CHECKING:
    from eitprocessing.datahandling.eitdata import EITData


def test_slicing(draeger1: Sequence):
    cutoff = 10
    data: EITData = draeger1

    assert data[cutoff] == data[cutoff]
    assert data[0:cutoff] == data[:cutoff]
    assert data[cutoff : len(data)] == data[cutoff:]

    assert Sequence.merge(data[:cutoff], data[cutoff:]) == data
    assert len(data[:cutoff]) == cutoff

    assert len(data) == len(data[cutoff:]) + len(data[-cutoff:])
    assert len(data) == len(data[:cutoff]) + len(data[:-cutoff])


def test_slicing2(
    draeger1: Sequence,
    timpel1: Sequence,
):
    cutoff = 100

    data: Sequence
    for data in [draeger1, timpel1]:
        assert data[0:cutoff] == data[:cutoff]
        assert data[cutoff : len(data)] == data[cutoff:]

        concatenated = Sequence.concatenate(data[:cutoff], data[cutoff:])
        concatenated.eit_data["raw"].path = data.eit_data["raw"].path
        assert concatenated == data
        assert len(data[:cutoff]) == cutoff

        assert len(data) == len(data[cutoff:]) + len(data[-cutoff:])
        assert len(data) == len(data[:cutoff]) + len(data[:-cutoff])


def test_select_by_time(
    draeger2: Sequence,
):
    # TODO (#82): this function is kinda ugly. Would be nice to refactor it
    # but I am struggling to think of a logical way to loop through.
    data = draeger2
    t22 = 55825.268
    t52 = 55826.768
    ms = 0.001

    # test illegal
    with pytest.warns(UserWarning):
        _ = data.select_by_time()
    with pytest.warns(UserWarning):
        _ = data.select_by_time(None, None)
    with pytest.warns(UserWarning):
        _ = data.select_by_time(None)
    with pytest.warns(UserWarning):
        _ = data.select_by_time(end=None)

    # test start only
    start_slices = [
        # (time, expectation if inclusive=True, expectation if inclusive=False)
        (t22, 22, 23),
        (t22 - ms, 22, 22),
        (t22 + ms, 23, 23),
    ]
    for start_slicing in start_slices:
        sliced = data.select_by_time(start=start_slicing[0], start_inclusive=True)
        assert len(sliced) == len(data) - start_slicing[1]
        sliced = data.select_by_time(start=start_slicing[0], start_inclusive=False)
        assert len(sliced) == len(data) - start_slicing[2]

    # test end only
    end_slices = [
        # (time, expectation if inclusive=True, expectation if inclusive=False)
        (t52, 52, 51),
        (t52 - ms, 51, 51),
        (t52 + ms, 52, 52),
    ]
    for end_slicing in end_slices:
        sliced = data.select_by_time(end=end_slicing[0], end_inclusive=True)
        assert len(sliced) == end_slicing[1]
        sliced = data.select_by_time(end=end_slicing[0], end_inclusive=False)
        assert len(sliced) == end_slicing[2]

    # test start and end
    for start_slicing in start_slices:
        for end_slicing in end_slices:
            # True/True
            sliced = data.select_by_time(
                start=start_slicing[0],
                end=end_slicing[0],
                start_inclusive=True,
                end_inclusive=True,
            )
            assert len(sliced) == end_slicing[1] - start_slicing[1]

            # False/True
            sliced = data.select_by_time(
                start=start_slicing[0],
                end=end_slicing[0],
                start_inclusive=False,
                end_inclusive=True,
            )
            assert len(sliced) == end_slicing[1] - start_slicing[2]

            # True/False
            sliced = data.select_by_time(
                start=start_slicing[0],
                end=end_slicing[0],
                start_inclusive=True,
                end_inclusive=False,
            )
            assert len(sliced) == end_slicing[2] - start_slicing[1]

            # False/False
            sliced = data.select_by_time(
                start=start_slicing[0],
                end=end_slicing[0],
                start_inclusive=False,
                end_inclusive=False,
            )
            assert len(sliced) == end_slicing[2] - start_slicing[2]


def test_concatenate_sequence(
    draeger1: Sequence,
    draeger2: Sequence,
    draeger_both: Sequence,
    timpel1: Sequence,
    timpel_double: Sequence,
):
    merged_draeger = Sequence.concatenate(draeger2, draeger1)
    assert len(merged_draeger.eit_data["raw"]) == len(draeger2.eit_data["raw"]) + len(
        draeger1.eit_data["raw"],
    )
    assert merged_draeger == draeger_both
    added_draeger = draeger2 + draeger1
    assert added_draeger == merged_draeger

    draeger_load_double = load_eit_data([draeger_file1, draeger_file1], "draeger")
    draeger_merge_double = Sequence.concatenate(draeger1, draeger1)
    assert draeger_load_double == draeger_merge_double
    added_draeger_double = draeger1 + draeger1
    assert added_draeger_double == draeger_merge_double

    draeger_merged_twice = Sequence.concatenate(draeger_merge_double, draeger_merge_double)
    draeger_load_four_times = load_eit_data([draeger_file1] * 4, "draeger")
    assert isinstance(draeger_merged_twice.path, list)
    assert len(draeger_merged_twice.path) == 4
    assert draeger_merged_twice == draeger_load_four_times

    draeger_merge_thrice = Sequence.concatenate(draeger_merge_double, draeger1)
    draeger_load_thrice = load_eit_data([draeger_file1] * 3, "draeger")
    assert isinstance(draeger_merge_thrice.eit_data.path, list)
    assert len(draeger_merge_thrice.path) == 3
    assert draeger_merge_thrice == draeger_load_thrice
    added_draeger_triple = draeger1 + draeger1 + draeger1
    assert draeger_merge_thrice == added_draeger_triple

    merged_timpel = Sequence.concatenate(timpel1, timpel1)
    assert len(merged_timpel) == 2 * len(timpel1)
    assert timpel_double == merged_timpel
    added_timpel = timpel1 + timpel1
    assert added_timpel == merged_timpel

    with pytest.raises(TypeError):
        _ = Sequence.concatenate(timpel1, draeger1)

    draeger1.framerate = 50
    with pytest.raises(ValueError):
        _ = Sequence.concatenate(draeger1, draeger2)

    draeger1.vendor = Vendor.TIMPEL
    with pytest.raises(ValueError):
        # TODO (#77): update this to AttributeError, once equivalence check for
        # framesets is implemented.
        _ = Sequence.concatenate(draeger1, timpel1)
