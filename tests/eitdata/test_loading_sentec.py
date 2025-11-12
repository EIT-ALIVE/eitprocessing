from pathlib import Path

import numpy as np
import pytest

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence


@pytest.mark.parametrize(
    "sequence",
    [
        "sentec_healthy_volunteer_1a",
        "sentec_healthy_volunteer_1b",
        "sentec_healthy_volunteer_2a",
        "sentec_healthy_volunteer_2b",
    ],
    indirect=["sequence"],
)
def test_load_sentec_single_file(
    sequence: Sequence,
):
    assert isinstance(sequence, Sequence)
    assert "raw" in sequence.eit_data

    eit_data = sequence.eit_data["raw"]

    assert isinstance(eit_data, EITData)
    assert np.isclose(eit_data.sample_frequency, 50.2, rtol=2e-2), "Sample frequency should be approximately 50.2 Hz"
    assert len(eit_data.time) > 0, "Time axis should not be empty"
    assert len(eit_data.time) == len(eit_data.pixel_impedance), "Length of time axis should match number of frames"
    assert len(eit_data) == len(eit_data.pixel_impedance), "Length of EITData should match number of frames"

    assert len(sequence.continuous_data) == 0, "Sentec data should not have continuous data channels"
    assert len(sequence.sparse_data) == 0, "Sentec data should not have sparse data channels"
    assert len(sequence.interval_data) == 0, "Sentec data should not have interval data channels"

    assert sequence == load_eit_data(sequence.eit_data["raw"].path, vendor="sentec", label=sequence.label), (
        "Loading with same parameters should yield same data"
    )

    sequence_loaded_w_sample_freq = load_eit_data(
        sequence.eit_data["raw"].path, vendor="sentec", sample_frequency=eit_data.sample_frequency, label=sequence.label
    )
    assert (
        sequence.eit_data["raw"].sample_frequency == sequence_loaded_w_sample_freq.eit_data["raw"].sample_frequency
    ), "When specifying the sample frequency, the final sample frequencies should match"
    assert sequence == sequence_loaded_w_sample_freq, "Loading providing the sample frequency should yield same data"

    assert sequence == load_eit_data(
        sequence.eit_data["raw"].path,
        vendor="sentec",
        sample_frequency=eit_data.sample_frequency,
        label="something else",
    ), "Loading with a different label should yield same data"


@pytest.mark.parametrize(
    ("sequence_a_fixture_name", "sequence_b_fixture_name", "sequence_merge_fixture_name"),
    [
        (
            "sentec_healthy_volunteer_1a",
            "sentec_healthy_volunteer_1b",
            "sentec_healthy_volunteer_1",
        ),
        (
            "sentec_healthy_volunteer_2a",
            "sentec_healthy_volunteer_2b",
            "sentec_healthy_volunteer_2",
        ),
    ],
)
def test_load_sentec_multiple_files(
    sequence_a_fixture_name: str,
    sequence_b_fixture_name: str,
    sequence_merge_fixture_name: str,
    request: pytest.FixtureRequest,
):
    sequence_a = request.getfixturevalue(sequence_a_fixture_name)
    sequence_b = request.getfixturevalue(sequence_b_fixture_name)
    sequence_merged = request.getfixturevalue(sequence_merge_fixture_name)

    assert len(sequence_merged) == len(sequence_a) + len(sequence_b), (
        "Combined length of individual sequences should match merged sequence"
    )
    assert sequence_merged == Sequence.concatenate(sequence_a, sequence_b), (
        "Merging individual sequences should equal pre-loaded merged sequence"
    )

    assert sequence_merged[: len(sequence_a)] == sequence_a, (
        "First part of merged sequence should match first individual sequence"
    )
    assert sequence_merged[len(sequence_a) :] == sequence_b, (
        "Second part of merged sequence should match second individual sequence"
    )


def test_load_sentec_skip_frames(sentec_healthy_volunteer_1a: Sequence, sentec_healthy_volunteer_1a_path: Path):
    n_frames = len(sentec_healthy_volunteer_1a)

    assert sentec_healthy_volunteer_1a == load_eit_data(
        sentec_healthy_volunteer_1a_path, vendor="sentec", first_frame=0
    )
    assert sentec_healthy_volunteer_1a == load_eit_data(
        sentec_healthy_volunteer_1a_path, vendor="sentec", max_frames=n_frames
    )
    assert sentec_healthy_volunteer_1a == load_eit_data(
        sentec_healthy_volunteer_1a_path, vendor="sentec", first_frame=0, max_frames=n_frames
    )

    first_frame = 100
    with pytest.warns(RuntimeWarning, match=r"The number of frames requested \(\d+\) is larger than"):
        sequence_skip_first = load_eit_data(
            sentec_healthy_volunteer_1a_path,
            vendor="sentec",
            first_frame=first_frame,
            max_frames=n_frames,
        )
    assert len(sequence_skip_first) == len(sentec_healthy_volunteer_1a) - first_frame, (
        "Loading from a later first_frame should yield fewer frames"
    )
    assert sequence_skip_first == sentec_healthy_volunteer_1a[first_frame:], (
        "Loaded sequence skipping first frames should match slicing"
    )

    max_frames = n_frames - 100
    sequence_limited_frames = load_eit_data(
        sentec_healthy_volunteer_1a_path,
        vendor="sentec",
        first_frame=0,
        max_frames=max_frames,
    )
    assert len(sequence_limited_frames) == max_frames, "Loading with max_frames should yield specified number of frames"
    assert sequence_limited_frames == sentec_healthy_volunteer_1a[:max_frames], (
        "Loaded sequence with limited frames should match slicing"
    )

    sequence_single_frame = load_eit_data(
        sentec_healthy_volunteer_1a_path,
        vendor="sentec",
        first_frame=n_frames - 1,
    )
    assert len(sequence_single_frame) == 1

    with pytest.raises(ValueError, match=r"`first_frame` \(\d+\) is larger than or equal to"):
        _ = load_eit_data(
            sentec_healthy_volunteer_1a_path,
            vendor="sentec",
            first_frame=n_frames,
        )
