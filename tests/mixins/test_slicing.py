import pytest

from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence


@pytest.mark.parametrize(
    "sequence",
    ["draeger_20hz_healthy_volunteer", "draeger_20hz_healthy_volunteer_pressure_pod"],
    indirect=True,
)
def test_slicing(sequence: Sequence):
    cutoff = 100
    assert len(sequence) > cutoff, "Test sequence should be longer than cutoff for meaningful test"

    assert sequence[0:cutoff] == sequence[:cutoff]
    assert sequence[cutoff : len(sequence)] == sequence[cutoff:]

    assert len(sequence[:cutoff]) == cutoff
    assert len(sequence) == len(sequence[cutoff:]) + len(sequence[-cutoff:])
    assert len(sequence) == len(sequence[:cutoff]) + len(sequence[:-cutoff])


@pytest.mark.parametrize("sequence", ["timpel_healthy_volunteer_1", "draeger_20hz_healthy_volunteer"], indirect=True)
def test_select_by_time(sequence: Sequence):
    pytest.skip("selecting by time not finalized yet")

    # test illegal
    with pytest.warns(UserWarning, match="No starting or end timepoints was selected"):
        _ = sequence.select_by_time()
    with pytest.warns(UserWarning, match="No starting or end timepoints was selected"):
        _ = sequence.select_by_time(None, None)
    with pytest.warns(UserWarning, match="No starting or end timepoints was selected"):
        _ = sequence.select_by_time(None)
    with pytest.warns(UserWarning, match="No starting or end timepoints was selected"):
        _ = sequence.select_by_time(end_time=None)

    # TODO (#82): this function is kinda ugly. Would be nice to refactor it
    # but I am struggling to think of a logical way to loop through.
    ms = 1 / 1000

    # test start_time only
    full_length = len(sequence)
    start_slices = [
        # (slice time, expected missing slices if inclusive=True, expected missing slices if inclusive=False)
        (sequence.time[22], 22, 22),
        (sequence.time[22] - ms, 21, 22),
        (sequence.time[22] + ms, 22, 23),
    ]
    for test_settings in start_slices:
        print(test_settings)  # noqa: T201
        sliced_inc = sequence.select_by_time(start_time=test_settings[0], start_inclusive=True)
        assert len(sliced_inc) == full_length - test_settings[1]
        sliced_exc = sequence.select_by_time(start_time=test_settings[0], start_inclusive=False)
        assert len(sliced_exc) == len(sequence) - test_settings[2]
        # test default:
        assert sequence.select_by_time(start_time=test_settings[0]) == sliced_inc

    # test end_time only
    end_slices = [
        # (slice time, expected length if inclusive=True, expected length if inclusive=False)
        (sequence.time[52], 52, 52),
        (sequence.time[52] - ms, 51, 52),
        (sequence.time[52] + ms, 52, 53),
    ]
    for test_settings in end_slices:
        print(test_settings)  # noqa: T201
        sliced_inc = sequence.select_by_time(end_time=test_settings[0], end_inclusive=True)
        assert len(sliced_inc) == test_settings[1]
        sliced_exc = sequence.select_by_time(end_time=test_settings[0], end_inclusive=False)
        assert len(sliced_exc) == test_settings[2]
        # test default:
        assert sequence.select_by_time(end_time=test_settings[0]) == sliced_exc

    # test start_time and end_time
    for start_slicing in start_slices:
        for end_slicing in end_slices:
            # True/True
            sliced = sequence.select_by_time(
                start_time=start_slicing[0],
                end_time=end_slicing[0],
                start_inclusive=True,
                end_inclusive=True,
            )
            assert len(sliced) == end_slicing[1] - start_slicing[1]

            # False/True
            sliced = sequence.select_by_time(
                start_time=start_slicing[0],
                end_time=end_slicing[0],
                start_inclusive=False,
                end_inclusive=True,
            )
            assert len(sliced) == end_slicing[1] - start_slicing[2]

            # True/False
            sliced = sequence.select_by_time(
                start_time=start_slicing[0],
                end_time=end_slicing[0],
                start_inclusive=True,
                end_inclusive=False,
            )
            assert len(sliced) == end_slicing[2] - start_slicing[1]

            # False/False
            sliced = sequence.select_by_time(
                start_time=start_slicing[0],
                end_time=end_slicing[0],
                start_inclusive=False,
                end_inclusive=False,
            )
            assert len(sliced) == end_slicing[2] - start_slicing[2]


def test_concatenate(
    draeger_20hz_healthy_volunteer: Sequence,
    draeger_20hz_healthy_volunteer_fixed_rr: Sequence,
    draeger_20hz_healthy_volunteer_and_fixed_rr: Sequence,
):
    """Tests concatenation against a sequence that is loaded from two files."""
    merged_sequence = Sequence.concatenate(draeger_20hz_healthy_volunteer, draeger_20hz_healthy_volunteer_fixed_rr)

    assert len(merged_sequence.eit_data["raw"]) == len(draeger_20hz_healthy_volunteer.eit_data["raw"]) + len(
        draeger_20hz_healthy_volunteer_fixed_rr.eit_data["raw"],
    ), "Length of concatenated sequence should equal sum of individual lengths."
    assert merged_sequence == draeger_20hz_healthy_volunteer_and_fixed_rr, (
        "Concatenated sequence should equal pre-loaded combined sequence."
    )

    added_sequence = draeger_20hz_healthy_volunteer + draeger_20hz_healthy_volunteer_fixed_rr
    assert added_sequence == merged_sequence, "Adding two sequences should be equivalent to concatenation."


def test_concatenate_three(
    draeger_20hz_healthy_volunteer: Sequence,
    draeger_20hz_healthy_volunteer_fixed_rr: Sequence,
    draeger_20hz_healthy_volunteer_pressure_pod: Sequence,
):
    merged_sequence_1 = Sequence.concatenate(draeger_20hz_healthy_volunteer, draeger_20hz_healthy_volunteer_fixed_rr)
    merged_sequence_2 = Sequence.concatenate(merged_sequence_1, draeger_20hz_healthy_volunteer_pressure_pod)

    paths = [
        sequence.eit_data["raw"].path
        for sequence in [
            draeger_20hz_healthy_volunteer,
            draeger_20hz_healthy_volunteer_fixed_rr,
            draeger_20hz_healthy_volunteer_pressure_pod,
        ]
    ]
    loaded_sequence = load_eit_data(paths, vendor="draeger")

    added_sequence = (
        draeger_20hz_healthy_volunteer
        + draeger_20hz_healthy_volunteer_fixed_rr
        + draeger_20hz_healthy_volunteer_pressure_pod
    )

    assert merged_sequence_2 == loaded_sequence, "Chained concatenation should equal files loaded together."
    assert merged_sequence_2 == added_sequence, "Chained addition should equal chained concatenation."


def test_merging_timing_order(
    draeger_20hz_healthy_volunteer_fixed_rr: Sequence, draeger_20hz_healthy_volunteer: Sequence
):
    with pytest.raises(
        ValueError, match=r"Concatenation failed. Second dataset \(.+\) may not start before first \(.+\) ends\."
    ):
        _ = Sequence.concatenate(draeger_20hz_healthy_volunteer_fixed_rr, draeger_20hz_healthy_volunteer)


@pytest.mark.parametrize("sequence", ["timpel_healthy_volunteer_1"], indirect=True)
def test_concatenate_slicing(sequence: Sequence):
    # slice and concatenate
    cutoff_point = 100
    part1 = sequence[:cutoff_point]
    part2 = sequence[cutoff_point:]
    assert sequence == Sequence.concatenate(part1, part2)

    # TODO: add tests for
    # - concatenating a third Sequence on top (or two double-sequences), also checking that path attribute is flat list
    # - as above, but for timpel and sentec


def test_concatenate_different_vendors(
    timpel_healthy_volunteer_1: Sequence,
    draeger_20hz_healthy_volunteer: Sequence,
):
    # Concatenate different vendors
    with pytest.raises(TypeError):
        _ = Sequence.concatenate(timpel_healthy_volunteer_1, draeger_20hz_healthy_volunteer)


def test_concatenate_different_sample_frequency(
    draeger_20hz_healthy_volunteer: Sequence,
    draeger_50hz_healthy_volunteer_pressure_pod: Sequence,
):
    with pytest.raises(ValueError):
        _ = Sequence.concatenate(draeger_20hz_healthy_volunteer, draeger_50hz_healthy_volunteer_pressure_pod)
