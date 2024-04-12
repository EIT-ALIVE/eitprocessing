import copy
import os
from pathlib import Path

import pytest

from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence

environment = os.environ.get(
    "EIT_PROCESSING_TEST_DATA",
    Path.resolve(Path(__file__).parent.parent),
)
data_directory = Path(environment) / "tests" / "test_data"
draeger_file1 = Path(data_directory) / "Draeger_Test3.bin"
draeger_file2 = Path(data_directory) / "Draeger_Test.bin"
timpel_file = Path(data_directory) / "Timpel_Test.txt"
dummy_file = Path(data_directory) / "not_a_file.dummy"


@pytest.fixture(scope="module")
def draeger_data1():
    return load_eit_data(draeger_file1, vendor="draeger", label="draeger1")


@pytest.fixture(scope="module")
def draeger_data2():
    return load_eit_data(draeger_file2, vendor="draeger", label="draeger2")


@pytest.fixture(scope="module")
def draeger_data_both():
    return load_eit_data([draeger_file2, draeger_file1], vendor="draeger", label="draeger_both")


@pytest.fixture(scope="module")
def timpel_data():
    return load_eit_data(timpel_file, vendor="timpel", label="timpel")


# @pytest.fixture()
# def timpel_data_double():
#     return load_eit_data([timpel_file, timpel_file], vendor="timpel", label="timpel_double")


def test_from_path_draeger(
    draeger_data1: Sequence,
    draeger_data2: Sequence,
    draeger_data_both: Sequence,
):
    assert isinstance(draeger_data1, Sequence)
    assert isinstance(draeger_data1.eit_data["raw"], EITData)
    assert draeger_data1.eit_data["raw"].framerate == 20  # noqa: PLR2004
    assert len(draeger_data1.eit_data["raw"]) == len(draeger_data1.eit_data["raw"].time)
    assert len(draeger_data2.eit_data["raw"].time) == 20740  # noqa: PLR2004

    assert draeger_data1 != draeger_data2

    # Load multiple
    assert len(draeger_data_both.eit_data["raw"]) == len(draeger_data1.eit_data["raw"]) + len(
        draeger_data2.eit_data["raw"],
    )

    # draeger_inverted = load_eit_data([draeger_file1, draeger_file2], vendor="draeger", label="inverted")
    # assert len(draeger_data_both) == len(draeger_inverted)
    # assert draeger_data_both != draeger_inverted


def test_from_path_timpel(
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
    # assert isinstance(timpel_data_double, Sequence)  #noqa: ERA001
    # assert len(timpel_data_double) == 2 * len(timpel_data)  #noqa: ERA001


def test_illegal_load_eit_data():
    # non existing
    for vendor in ["draeger", "timpel"]:
        with pytest.raises(FileNotFoundError):
            _ = load_eit_data(dummy_file, vendor=vendor)

    # incorrect vendor
    with pytest.raises(OSError):
        _ = load_eit_data(draeger_file1, vendor="timpel")
    with pytest.raises(OSError):
        _ = load_eit_data(timpel_file, vendor="draeger")


def test_merge(
    draeger_data1: Sequence,
    draeger_data2: Sequence,
    draeger_data_both: Sequence,
    timpel_data: Sequence,
    timpel_data_double: Sequence,
):
    merged_draeger = Sequence.concatenate(draeger_data2, draeger_data1)
    assert len(merged_draeger.eit_data["raw"]) == len(draeger_data2.eit_data["raw"]) + len(
        draeger_data1.eit_data["raw"],
    )
    assert merged_draeger == draeger_data_both
    added_draeger = draeger_data2 + draeger_data1
    assert added_draeger == merged_draeger

    draeger_load_double = load_eit_data([draeger_file1, draeger_file1], "draeger")
    draeger_merge_double = Sequence.merge(draeger_data1, draeger_data1)
    assert draeger_load_double == draeger_merge_double
    added_draeger_double = draeger_data1 + draeger_data1
    assert added_draeger_double == draeger_merge_double

    draeger_merged_twice = Sequence.merge(draeger_merge_double, draeger_merge_double)
    draeger_load_four_times = load_eit_data([draeger_file1] * 4, "draeger")
    assert isinstance(draeger_merged_twice.path, list)
    assert len(draeger_merged_twice.path) == 4  # noqa: PLR2004
    assert draeger_merged_twice == draeger_load_four_times

    draeger_merge_thrice = Sequence.merge(draeger_merge_double, draeger_data1)
    draeger_load_thrice = load_eit_data([draeger_file1] * 3, "draeger")
    assert isinstance(draeger_merge_thrice.path, list)
    assert len(draeger_merge_thrice.path) == 3  # noqa: PLR2004
    assert draeger_merge_thrice == draeger_load_thrice
    added_draeger_triple = draeger_data1 + draeger_data1 + draeger_data1
    assert draeger_merge_thrice == added_draeger_triple

    merged_timpel = Sequence.merge(timpel_data, timpel_data)
    assert len(merged_timpel) == 2 * len(timpel_data)
    assert timpel_data_double == merged_timpel
    added_timpel = timpel_data + timpel_data
    assert added_timpel == merged_timpel

    with pytest.raises(TypeError):
        _ = Sequence.merge(timpel_data, draeger_data1)

    draeger_data1.framerate = 50
    with pytest.raises(ValueError):
        _ = Sequence.merge(draeger_data1, draeger_data2)

    draeger_data1.vendor = Vendor.TIMPEL
    with pytest.raises(ValueError):
        # TODO (#77): update this to AttributeError, once equivalence check for
        # framesets is implemented.
        _ = Sequence.merge(draeger_data1, timpel_data)


def test_copy(
    draeger_data1: Sequence,
    timpel_data: Sequence,
):
    data: Sequence
    for data in [draeger_data1, timpel_data]:
        data_copy = copy.deepcopy(data)
        assert data == data_copy


def test_equals(
    draeger_data1: Sequence,
    timpel_data: Sequence,
):
    data: Sequence
    for data in [draeger_data1, timpel_data]:
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
    draeger_data1: Sequence,
    timpel_data: Sequence,
):
    cutoff = 100

    data: Sequence
    for data in [draeger_data1, timpel_data]:
        assert data[0:cutoff] == data[:cutoff]
        assert data[cutoff : len(data)] == data[cutoff:]

        concatenated = Sequence.concatenate(data[:cutoff], data[cutoff:])
        concatenated.eit_data["raw"].path = data.eit_data["raw"].path
        assert concatenated == data
        assert len(data[:cutoff]) == cutoff

        assert len(data) == len(data[cutoff:]) + len(data[-cutoff:])
        assert len(data) == len(data[:cutoff]) + len(data[:-cutoff])


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
    timpel_first_part = load_eit_data(timpel_file, "timpel", max_frames=cutoff)
    timpel_second_part = load_eit_data(timpel_file, "timpel", first_frame=cutoff)

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


def test_illegal_first():
    for ff in [0.5, -1, "fdw"]:
        with pytest.raises((TypeError, ValueError)):
            _ = load_eit_data(draeger_file1, "draeger", first_frame=ff)

    for ff2 in [0, 0.0, 1.0, None]:
        _ = load_eit_data(draeger_file1, "draeger", first_frame=ff2)


def test_select_by_time(
    draeger_data2: Sequence,
):
    # TODO (#82): this function is kinda ugly. Would be nice to refactor it
    # but I am struggling to think of a logical way to loop through.
    data = draeger_data2
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


def test_label(
    draeger_data1: Sequence,
    draeger_data2: Sequence,
):
    assert isinstance(draeger_data1.label, str), "default label is not a string"
    assert draeger_data1.label == f"Sequence_{id(draeger_data1)}", "unexpected default label"

    assert draeger_data1.label != draeger_data2.label, "different data has identical label"

    timpel_1 = load_eit_data(timpel_file, vendor="timpel")
    timpel_2 = load_eit_data(timpel_file, vendor="timpel")
    assert timpel_1.label != timpel_2.label, "reloaded data has identical label"

    test_label = "test_label"
    timpel_3 = load_eit_data(timpel_file, vendor="timpel", label=test_label)
    timpel_4 = load_eit_data(timpel_file, vendor="timpel", label=test_label)
    assert timpel_3.label == test_label, "label attribute does not match given label"
    assert timpel_3.label == timpel_4.label, "re-used test label not recognized as identical"

    timpel_copy = timpel_1.deepcopy()
    assert timpel_1.label != timpel_copy.label, "deepcopied data has identical label"
    assert timpel_copy.label == f"Copy of <{timpel_1.label}>", "deepcopied data has unexpected label"
    timpel_copy_relabel = timpel_1.deepcopy(label=test_label)
    assert timpel_1.label != timpel_copy_relabel.label, "deepcopied data with new label has identical label"
    timpel_copy_relabel = timpel_1.deepcopy(relabel=False)
    assert timpel_1.label == timpel_copy_relabel.label, "deepcopied data did not keep old label"
    timpel_copy_relabel = timpel_1.deepcopy(label=test_label, relabel=False)
    assert timpel_1.label != timpel_copy_relabel.label, "combo of label and relabel not working as intended"


def test_relabeling(
    timpel_data: Sequence,
    draeger_data2: Sequence,
):
    test_label = "test label"

    # merging
    merged_timpel = Sequence.merge(timpel_data, timpel_data)
    assert merged_timpel.label != timpel_data.label, "merging does not assign new label by default"
    assert (
        merged_timpel.label == f"Merge of <{timpel_data.label}> and <{timpel_data.label}>"
    ), "merging generates unexpected default label"
    added_timpel = timpel_data + timpel_data
    assert (
        added_timpel.label == f"Merge of <{timpel_data.label}> and <{timpel_data.label}>"
    ), "adding generates unexpected default label"
    merged_timpel_2 = Sequence.merge(timpel_data, timpel_data, label=test_label)
    assert merged_timpel_2.label == test_label, "incorrect label assigned when merging data with new label"

    # slicing
    indices = slice(0, 10)
    sliced_timpel = timpel_data[indices]
    assert sliced_timpel.label != timpel_data.label, "slicing does not assign new label by default"
    assert (
        sliced_timpel.label == f"Slice ({indices.start}-{indices.stop}) of <{timpel_data.label}>"
    ), "slicing generates unexpected default label"
    sliced_timpel_2 = timpel_data.select_by_index(indices=indices, label=test_label)
    assert sliced_timpel_2.label == test_label, "incorrect label assigned when slicing data with new label"

    # select_by_time
    t22 = 55825.268
    t52 = 55826.768
    time_sliced = draeger_data2.select_by_time(t22, t52 + 0.001)
    assert time_sliced.label != draeger_data2.label, "time slicing does not assign new label by default"
    assert (
        time_sliced.label == f"Slice (22-52) of <{draeger_data2.label}>"
    ), "slicing generates unexpected default label"
    time_sliced_2 = draeger_data2.select_by_time(t22, t52, label=test_label)
    assert time_sliced_2.label == test_label, "incorrect label assigned when time slicing data with new label"
