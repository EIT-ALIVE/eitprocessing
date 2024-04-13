import pytest  # noqa: F401 (needed for fixtures)

from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence
from tests.conftest import timpel_file


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
    merged_timpel = Sequence.concatenate(timpel_data, timpel_data)
    assert merged_timpel.label != timpel_data.label, "merging does not assign new label by default"
    assert (
        merged_timpel.label == f"Merge of <{timpel_data.label}> and <{timpel_data.label}>"
    ), "merging generates unexpected default label"
    added_timpel = timpel_data + timpel_data
    assert (
        added_timpel.label == f"Merge of <{timpel_data.label}> and <{timpel_data.label}>"
    ), "adding generates unexpected default label"
    merged_timpel_2 = Sequence.concatenate(timpel_data, timpel_data, label=test_label)
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
