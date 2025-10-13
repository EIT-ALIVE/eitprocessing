from pathlib import Path

import pytest

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence


@pytest.mark.parametrize(
    ("sequence_fixture_name", "data_path_fixture_name", "length"),
    [
        ("draeger_porcine_1", "draeger_porcine_1_path", 14140),
        ("draeger_porcine_2", "draeger_porcine_2_path", 11840),
    ],
)
def test_load_draeger_porcine(
    request: pytest.FixtureRequest, sequence_fixture_name: str, data_path_fixture_name: str, length: int
):
    sequence = request.getfixturevalue(sequence_fixture_name)
    data_path = request.getfixturevalue(data_path_fixture_name)

    assert isinstance(sequence, Sequence), "Loaded object should be a Sequence"
    assert isinstance(sequence.eit_data["raw"], EITData), "Sequence should contain EITData with 'raw' key"
    assert sequence.eit_data["raw"].path == data_path
    assert sequence.eit_data["raw"].sample_frequency == 20, "Sample frequency should be 20 Hz"
    assert len(sequence.eit_data["raw"]) == len(sequence.eit_data["raw"].time), (
        "Length of EITData should match length of time axis"
    )
    assert len(sequence.eit_data["raw"].time) == length, f"{sequence.label} should contain 14140 frames"

    assert len(sequence.continuous_data) == 6 + 1, (
        "Draeger data should have 6 continuous medibus fields + the calculated global impedance"
    )

    assert sequence == load_eit_data(data_path, vendor="draeger", sample_frequency=20, label=sequence.label), (
        "Loading with same parameters should yield same data"
    )
    assert sequence == load_eit_data(data_path, vendor="draeger", sample_frequency=20, label="something_else"), (
        "Loading with different label should yield same data"
    )


def test_draeger_porcine_1_2_differ(draeger_porcine_1: Sequence, draeger_porcine_2: Sequence):
    assert draeger_porcine_1 != draeger_porcine_2, "Different files should yield different data"


def test_draeger_porcine_1_and_2(
    draeger_porcine_1: Sequence, draeger_porcine_2: Sequence, draeger_porcine_1_and_2: Sequence
):
    # Load multiple
    assert len(draeger_porcine_1_and_2) == len(draeger_porcine_1) + len(draeger_porcine_2), (
        "Combined data length should equal sum of individual lengths"
    )


@pytest.mark.parametrize(
    "data_path_fixture_name",
    ["draeger_porcine_1_path", "draeger_porcine_2_path"],
)
def test_draeger_sample_frequency(request: pytest.FixtureRequest, data_path_fixture_name: str):
    data_path = request.getfixturevalue(data_path_fixture_name)
    with_sf = load_eit_data(data_path, vendor="draeger", sample_frequency=20)
    without_sf = load_eit_data(data_path, vendor="draeger")
    assert with_sf.eit_data["raw"].sample_frequency == without_sf.eit_data["raw"].sample_frequency


def test_draeger_sample_frequency_mismatch_warning(draeger_porcine_1_path: Path):
    with pytest.warns(RuntimeWarning, match="Provided sample frequency"):
        _ = load_eit_data(draeger_porcine_1_path, vendor="draeger", sample_frequency=25)
