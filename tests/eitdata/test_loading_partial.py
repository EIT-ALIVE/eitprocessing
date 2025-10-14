import pytest

from eitprocessing.datahandling.loading import load_eit_data


# TODO: add other vendors
# TODO: add dataset with events, and test loading from the frame at or just after the event
@pytest.mark.parametrize(
    ("data_path_fixture_name", "sequence_fixture_name", "split_frame", "vendor", "sample_frequency"),
    [("draeger_porcine_1_path", "draeger_porcine_1", 100, "draeger", 20)],
)
def test_load_partial(
    data_path_fixture_name: str,
    sequence_fixture_name: str,
    split_frame: int,
    vendor: str,
    sample_frequency: float,
    request: pytest.FixtureRequest,
):
    data_path = request.getfixturevalue(data_path_fixture_name)
    sequence_full = request.getfixturevalue(sequence_fixture_name)

    sequence_part_1 = load_eit_data(
        data_path, vendor=vendor, sample_frequency=sample_frequency, max_frames=split_frame, label="part 1"
    )

    sequence_part_2 = load_eit_data(
        data_path, vendor=vendor, sample_frequency=sample_frequency, first_frame=split_frame, label="part 2"
    )

    assert len(sequence_part_1) == split_frame, "The first sequence should contain the specified number of frames"
    assert len(sequence_part_2) == len(sequence_full) - split_frame, (
        "The second sequence should contains the remaining frames"
    )
    assert len(sequence_part_1) + len(sequence_part_2) == len(sequence_full), (
        "The combined length should match the total length"
    )

    assert sequence_part_1 == sequence_full[:split_frame], "The first part should match the beginning of the full data"

    assert sequence_part_1.concatenate(sequence_part_2) == sequence_full, (
        "Concatenating both parts should reconstruct the full data"
    )

    # TODO: enable after fixing select_by_time issues
    pytest.skip("Currently fails due to select_by_time issues")
    assert sequence_part_2 == sequence_full[split_frame:], "The second part should match the end of the full data"
