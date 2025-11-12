from pathlib import Path

import pytest  # TODO: noqa: F401 (needed for fixtures) once the pytest.skip is removed

from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence


def test_default_label(
    draeger_20hz_healthy_volunteer_path: Path,
    draeger_20hz_healthy_volunteer: Sequence,
    timpel_healthy_volunteer_1_path: Path,
):
    draeger_default = load_eit_data(draeger_20hz_healthy_volunteer_path, vendor="draeger", sample_frequency=20)
    assert isinstance(draeger_default.label, str)
    assert draeger_default.label == f"Sequence_{id(draeger_default)}"

    timpel_default = load_eit_data(timpel_healthy_volunteer_1_path, vendor="timpel")
    assert isinstance(timpel_default.label, str)
    assert timpel_default.label == f"Sequence_{id(timpel_default)}"

    # test that default label changes upon reloading identical data
    draeger_reloaded = load_eit_data(draeger_20hz_healthy_volunteer_path, vendor="draeger", sample_frequency=20)
    assert draeger_default == draeger_reloaded
    assert draeger_default.label != draeger_reloaded.label
    assert draeger_default.label != draeger_20hz_healthy_volunteer.label


def test_relabeling(
    timpel_healthy_volunteer_1: Sequence,
    draeger_20hz_healthy_volunteer_fixed_rr: Sequence,
    draeger_20hz_healthy_volunteer: Sequence,
):
    pytest.skip("changing labels is currently bugging")
    # merging
    merged = Sequence.concatenate(draeger_20hz_healthy_volunteer_fixed_rr, draeger_20hz_healthy_volunteer)
    assert merged.label != draeger_20hz_healthy_volunteer.label
    assert merged.label != draeger_20hz_healthy_volunteer_fixed_rr.label
    assert (
        merged.label
        == f"Merge of <{draeger_20hz_healthy_volunteer_fixed_rr.label}> and <{draeger_20hz_healthy_volunteer.label}>"
    )

    # slicing
    indices = slice(3, 12)
    sliced_timpel = timpel_healthy_volunteer_1[indices]
    assert sliced_timpel.label != timpel_healthy_volunteer_1.label
    assert sliced_timpel.label == f"Slice ({indices.start}-{indices.stop}] of <{timpel_healthy_volunteer_1.label}>"

    # custom new label:)
    test_label = "test label"
    merged = Sequence.concatenate(
        draeger_20hz_healthy_volunteer_fixed_rr, draeger_20hz_healthy_volunteer, newlabel=test_label
    )
    assert merged.label == test_label
    sliced_timpel = Sequence.select_by_index(
        timpel_healthy_volunteer_1, start=indices.start, end=indices.stop, newlabel=test_label
    )
    assert sliced_timpel.label == test_label

    # selecting by time
    pytest.skip("selecting by time not finalized yet")
    t22 = 55825.268
    t52 = 55826.768
    time_sliced = draeger_20hz_healthy_volunteer_fixed_rr.select_by_time(t22, t52 + 0.001)
    assert time_sliced.label != draeger_20hz_healthy_volunteer_fixed_rr.label, (
        "time slicing does not assign new label by default"
    )
    assert time_sliced.label == f"Slice (22-52) of <{draeger_20hz_healthy_volunteer_fixed_rr.label}>", (
        "slicing generates unexpected default label"
    )
    time_sliced_2 = draeger_20hz_healthy_volunteer_fixed_rr.select_by_time(t22, t52, label=test_label)
    assert time_sliced_2.label == test_label, "incorrect label assigned when time slicing data with new label"
