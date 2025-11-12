import tempfile
from pathlib import Path

import numpy as np
import pytest

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.loading.draeger import _bin_file_formats
from eitprocessing.datahandling.sequence import Sequence


# TODO: create/find data with 6 continuous data channels
@pytest.mark.parametrize(
    ("sequence", "sequence_path", "length", "n_continuous_channels", "sample_frequency"),
    [
        ("draeger_20hz_healthy_volunteer", "draeger_20hz_healthy_volunteer_path", 6920, 10, 20),
        ("draeger_20hz_healthy_volunteer_fixed_rr", "draeger_20hz_healthy_volunteer_fixed_rr_path", 7340, 10, 20),
        (
            "draeger_20hz_healthy_volunteer_pressure_pod",
            "draeger_20hz_healthy_volunteer_pressure_pod_path",
            1320,
            10,
            20,
        ),
        (
            "draeger_50hz_healthy_volunteer_pressure_pod",
            "draeger_50hz_healthy_volunteer_pressure_pod_path",
            3700,
            10,
            50,
        ),
        (
            "draeger_20hz_healthy_volunteer_time_wrap_v120",
            "draeger_20hz_healthy_volunteer_time_wrap_v120_path",
            2460,
            6,
            20,
        ),
        (
            "draeger_20hz_healthy_volunteer_time_wrap_v130",
            "draeger_20hz_healthy_volunteer_time_wrap_v130_path",
            2460,
            10,
            20,
        ),
    ],
    indirect=["sequence", "sequence_path"],
)
def test_load_draeger(
    sequence: Sequence,
    sequence_path: Path,
    length: int,
    n_continuous_channels: int,
    sample_frequency: float,
):
    assert isinstance(sequence, Sequence), "Loaded object should be a Sequence"
    assert isinstance(sequence.eit_data["raw"], EITData), "Sequence should contain EITData with 'raw' key"
    assert sequence.eit_data["raw"].path == sequence_path
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
        sequence_path, vendor="draeger", sample_frequency=sample_frequency, label=sequence.label
    ), "Loading with same parameters should yield same data"
    assert sequence == load_eit_data(
        sequence_path, vendor="draeger", sample_frequency=sample_frequency, label="something_else"
    ), "Loading with different label should yield same data"
    assert sequence == load_eit_data(sequence_path, vendor="draeger"), (
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


def test_events(draeger_20hz_healthy_volunteer: Sequence):
    events = draeger_20hz_healthy_volunteer.sparse_data["events_(draeger)"]
    assert len(events) == 1, "There should be 1 event in the draeger_20hz_healthy_volunteer data"
    assert events.values[0].text == "bed inflating", "Event text should match expected value"


def test_event_on_first_frame(draeger_20hz_healthy_volunteer: Sequence):
    """Tests loading a sequence where there is an event on the first frame.

    There are two ways this can occur. The first is when the frame occurs later in the file, but we skip all frames
    before the first frame with an event. This is tested by loading with first_frame set to the index of the first event
    (`seq_first_frame_is_event_index`). The second way this can happen is if the data file has an event on the first
    frame. We test this by creating a temporary copy of the data file, where the initial frames before the event file
    are removed.
    """
    event_index = np.searchsorted(
        draeger_20hz_healthy_volunteer.time,
        draeger_20hz_healthy_volunteer.sparse_data["events_(draeger)"].time[0],
    )

    seq_first_frame_is_event_index = load_eit_data(
        draeger_20hz_healthy_volunteer.eit_data["raw"].path, vendor="draeger", first_frame=event_index
    )
    assert (
        np.searchsorted(
            seq_first_frame_is_event_index.time,
            seq_first_frame_is_event_index.data["events_(draeger)"].time[0],
        )
        == 0
    ), "The event should be on the first frame when loading from its timepoint."

    frame_size = _bin_file_formats["pressure_pod"]["frame_size"]
    ignore_bytes = event_index * frame_size  # number of bytes to ignore at start of file

    with tempfile.NamedTemporaryFile(delete_on_close=False) as temporary_file:
        # Create a temporary file, that is removed after the context manager is closed
        tempfile_path = Path(temporary_file.name)
        with draeger_20hz_healthy_volunteer.eit_data["raw"].path.open("rb") as original_file:
            original_file.seek(ignore_bytes)  # skip frames before the event
            temporary_file.write(original_file.read())  # write remaining data to temp file

        seq_trimmed_file = load_eit_data(tempfile_path, vendor="draeger")

    assert seq_first_frame_is_event_index == seq_trimmed_file, (
        "Loading from temp file should match loading from original file skipping frames."
    )
