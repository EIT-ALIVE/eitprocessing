from pathlib import Path

import pytest

from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence


# TODO: add other vendors
# TODO: add dataset with events, and test loading from the frame at or just after the event
@pytest.mark.parametrize(
    ("sequence_path", "sequence", "split_frame", "vendor", "sample_frequency"),
    [
        ("draeger_20hz_healthy_volunteer_path", "draeger_20hz_healthy_volunteer", 100, "draeger", 20),
        ("timpel_healthy_volunteer_1_path", "timpel_healthy_volunteer_1", 100, "timpel", None),
    ],
    indirect=["sequence_path", "sequence"],
)
def test_load_partial(
    sequence_path: Path,
    sequence: Sequence,
    split_frame: int,
    vendor: str,
    sample_frequency: float,
):
    sequence_part_1 = load_eit_data(
        sequence_path, vendor=vendor, sample_frequency=sample_frequency, max_frames=split_frame, label="part 1"
    )

    sequence_part_2 = load_eit_data(
        sequence_path, vendor=vendor, sample_frequency=sample_frequency, first_frame=split_frame, label="part 2"
    )

    assert len(sequence_part_1) == split_frame, "The first sequence should contain the specified number of frames"
    assert len(sequence_part_2) == len(sequence) - split_frame, (
        "The second sequence should contains the remaining frames"
    )
    assert len(sequence_part_1) + len(sequence_part_2) == len(sequence), (
        "The combined length should match the total length"
    )

    assert sequence_part_1 == sequence[:split_frame], "The first part should match the beginning of the full data"

    assert sequence_part_1.concatenate(sequence_part_2) == sequence, (
        "Concatenating both parts should reconstruct the full data"
    )

    # TODO: enable after fixing select_by_time issues
    pytest.skip("Currently fails due to select_by_time issues")
    assert sequence_part_2 == sequence[split_frame:], "The second part should match the end of the full data"
