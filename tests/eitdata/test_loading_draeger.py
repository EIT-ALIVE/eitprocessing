from pathlib import Path

import pytest

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence


# TODO: create/find data with 6 continuous data channels
@pytest.mark.parametrize(
    ("sequence_fixture_name", "length", "n_continuous_channels", "sample_frequency"),
    [
        ("draeger_20hz_healthy_volunteer", 6920, 10, 20),
        ("draeger_20hz_healthy_volunteer_fixed_rr", 7340, 10, 20),
        ("draeger_20hz_healthy_volunteer_pressure_pod", 1320, 10, 20),
        ("draeger_50hz_healthy_volunteer_pressure_pod", 3700, 10, 50),
    ],
)
def test_load_draeger_porcine(
    request: pytest.FixtureRequest,
    sequence_fixture_name: str,
    length: int,
    n_continuous_channels: int,
    sample_frequency: float,
):
    sequence = request.getfixturevalue(sequence_fixture_name)
    data_path = request.getfixturevalue(f"{sequence_fixture_name}_path")

    assert isinstance(sequence, Sequence), "Loaded object should be a Sequence"
    assert isinstance(sequence.eit_data["raw"], EITData), "Sequence should contain EITData with 'raw' key"
    assert sequence.eit_data["raw"].path == data_path
    assert sequence.eit_data["raw"].sample_frequency == sample_frequency, (
        f"Sample frequency should be {sample_frequency:.1f} Hz"
    )
    assert len(sequence.eit_data["raw"]) == len(sequence.eit_data["raw"].time), (
        "Length of EITData should match length of time axis"
    )
    assert len(sequence.eit_data["raw"].time) == length, f"{sequence.label} should contain 14140 frames"

    assert len(sequence.continuous_data) == n_continuous_channels + 1, (
        "Draeger data should have 6 continuous medibus fields + the calculated global impedance"
    )

    assert sequence == load_eit_data(
        data_path, vendor="draeger", sample_frequency=sample_frequency, label=sequence.label
    ), "Loading with same parameters should yield same data"
    assert sequence == load_eit_data(
        data_path, vendor="draeger", sample_frequency=sample_frequency, label="something_else"
    ), "Loading with different label should yield same data"
    assert sequence == load_eit_data(data_path, vendor="draeger"), (
        "Loading without sample frequency should yield the same data"
    )


def test_draeger_20hz_healthy_volunteer_2_differ(
    draeger_20hz_healthy_volunteer: Sequence, draeger_20hz_healthy_volunteer_fixed_rr: Sequence
):
    assert draeger_20hz_healthy_volunteer != draeger_20hz_healthy_volunteer_fixed_rr, (
        "Different files should yield different data"
    )


def test_draeger_20hz_healthy_volunteer_and_fixed_rr(
    draeger_20hz_healthy_volunteer: Sequence,
    draeger_20hz_healthy_volunteer_fixed_rr: Sequence,
    draeger_20hz_healthy_volunteer_and_fixed_rr: Sequence,
):
    # Load multiple
    assert len(draeger_20hz_healthy_volunteer_and_fixed_rr) == len(draeger_20hz_healthy_volunteer) + len(
        draeger_20hz_healthy_volunteer_fixed_rr
    ), "Combined data length should equal sum of individual lengths"


def test_draeger_sample_frequency_mismatch_warning(draeger_20hz_healthy_volunteer_path: Path):
    with pytest.warns(RuntimeWarning, match="Provided sample frequency"):
        _ = load_eit_data(draeger_20hz_healthy_volunteer_path, vendor="draeger", sample_frequency=25)


def test_estimate_sample_frequency_few_points(draeger_20hz_healthy_volunteer_path: Path):
    with pytest.raises(ValueError, match="Could not estimate sample frequency from time axis"):
        _ = load_eit_data(draeger_20hz_healthy_volunteer_path, vendor="draeger", max_frames=1)

    with pytest.warns(RuntimeWarning, match="Could not estimate sample frequency from time axis"):
        _ = load_eit_data(draeger_20hz_healthy_volunteer_path, vendor="draeger", max_frames=1, sample_frequency=20)

    without_sf = load_eit_data(draeger_20hz_healthy_volunteer_path, vendor="draeger", max_frames=2)
    with_sf = load_eit_data(draeger_20hz_healthy_volunteer_path, vendor="draeger", sample_frequency=20, max_frames=2)
    assert without_sf == with_sf, "Loading without provided sample frequency should work with few data points"


def test_skipping_frames(draeger_20hz_healthy_volunteer: Sequence):
    n_frames = len(draeger_20hz_healthy_volunteer)

    assert draeger_20hz_healthy_volunteer == load_eit_data(
        draeger_20hz_healthy_volunteer.eit_data["raw"].path, vendor="draeger", first_frame=0
    )
    assert draeger_20hz_healthy_volunteer == load_eit_data(
        draeger_20hz_healthy_volunteer.eit_data["raw"].path, vendor="draeger", max_frames=n_frames
    )

    short_sequence_1 = load_eit_data(
        draeger_20hz_healthy_volunteer.eit_data["raw"].path, vendor="draeger", first_frame=n_frames - 2
    )
    assert len(short_sequence_1) == 2, "Loading from near end should yield 2 frames"

    short_sequence_2 = load_eit_data(
        draeger_20hz_healthy_volunteer.eit_data["raw"].path, vendor="draeger", max_frames=2
    )
    assert len(short_sequence_2) == 2, "Loading with max_frames=2 should yield 2 frames"

    with pytest.warns(
        RuntimeWarning,
        match=r"The number of frames requested \(\d+\) is larger than the available number \(\d+\) of frames.",
    ):
        _ = load_eit_data(
            draeger_20hz_healthy_volunteer.eit_data["raw"].path, vendor="draeger", max_frames=n_frames + 1
        )

    with pytest.raises(ValueError, match="No frames to load with"):
        _ = load_eit_data(draeger_20hz_healthy_volunteer.eit_data["raw"].path, vendor="draeger", max_frames=0)
