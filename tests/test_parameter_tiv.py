import copy
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.parameters.tidal_impedance_variation import TIV

environment = Path(
    os.environ.get(
        "EIT_PROCESSING_TEST_DATA",
        Path(__file__).parent.parent.resolve(),
    ),
)
data_directory = environment / "tests" / "test_data"
draeger_file1 = data_directory / "Draeger_Test3.bin"
timpel_file = data_directory / "Timpel_Test.txt"


class MockEITData(EITData):
    """Class to create Mock EITData for running tests."""

    def __init__(self):
        """Mock pixel_impedance.

        Each pixel is an identical sine wave with a baseline drop to create difference between
        inspiratory and expiratory limb for one of the breaths.
        """
        duration = 10
        fs = 1000
        frequency = 0.25
        amplitude = 1.0
        baseline_drop = -0.5
        # Calculate the period of the sine wave
        period = 1 / frequency

        # Create time vector with extra breaths
        extra_breaths_duration = 2 * period  # One period for the extra breath at start and end
        total_duration = duration + extra_breaths_duration
        t = np.linspace(0, total_duration, int(total_duration * fs), endpoint=False)

        # Generate the sine wave
        signal = amplitude * np.sin(2 * np.pi * frequency * t)

        # Apply baseline drop halfway through the original duration
        halfway_index = len(signal) // 2
        signal[halfway_index:] += baseline_drop  # Shift the baseline down for the second half

        # Store the waves in a (time, 2, 2) array
        pixel_impedance = np.empty((len(t), 2, 2))
        pixel_impedance[:, 0, 0] = signal
        pixel_impedance[:, 0, 1] = signal
        pixel_impedance[:, 1, 0] = signal
        pixel_impedance[:, 1, 1] = signal

        self.time = t
        self.pixel_impedance = pixel_impedance
        self.label = "raw"


class MockContinuousData(ContinuousData):
    """Class to create Mock ContinuousData for running tests."""

    def __init__(self, mock_eit_data: MockEITData):
        pixel_impedance = mock_eit_data.pixel_impedance
        self.time = mock_eit_data.time
        self.values = np.nansum(
            pixel_impedance,
            axis=(1, 2),
        )
        self.label = "global_impedance(raw)"
        self.sample_frequency = 1000


class MockIntervalData:
    """Class to create Mock IntervalData for running tests."""

    def __init__(self):
        self.data = []

    def add(self, item: list):
        """Function to add item to IntervalData."""
        self.data.append(item)


class MockSequence(Sequence):
    """Class to create Mock Sequence for running tests."""

    def __init__(self, mock_eit_data: MockEITData, mock_continuous_data: MockContinuousData):
        self.eit_data = mock_eit_data
        self.continuous_data = mock_continuous_data
        self.interval_data = MockIntervalData()


@pytest.fixture()
def mock_continuous_data(mock_eit_data: MockEITData):
    """Fixture to provide an instance of MockContinuousData."""
    return MockContinuousData(mock_eit_data)


@pytest.fixture()
def mock_eit_data():
    """Fixture to provide an instance of MockEITData."""
    return MockEITData()


@pytest.fixture()
def mock_sequence(mock_eit_data: MockEITData, mock_continuous_data: MockEITData):
    """Fixture to provide an instance of MockSequence."""
    return MockSequence(mock_eit_data, mock_continuous_data)


def test_tiv_initialization():
    """Test that TIV initializes correctly with default parameters."""
    tiv = TIV()
    assert tiv.method == "extremes"
    assert tiv.breath_detection_kwargs == {}


def test_compute_parameter_type_error():
    """Test that compute_parameter raises TypeError for unsupported data types."""
    tiv = TIV()

    with pytest.raises(TypeError, match="This method is implemented for ContinuousData or EITData"):
        # Pass an unsupported type to trigger TypeError
        tiv.compute_parameter("unsupported_type")


def test_compute_continuous_parameter_not_implemented_method(mock_continuous_data: MockContinuousData):
    """Test that compute_continuous_parameter raises NotImplementedError for unsupported methods."""
    tiv = TIV(method="unsupported_method")
    with pytest.raises(NotImplementedError, match="Method unsupported_method is not implemented."):
        tiv.compute_continuous_parameter(mock_continuous_data)


def test_compute_pixel_parameter_not_implemented_method(
    mock_eit_data: MockEITData,
    mock_continuous_data: MockContinuousData,
    mock_sequence: MockSequence,
):
    """Test that compute_pixel_parameter raises NotImplementedError for unsupported methods."""
    tiv = TIV(method="unsupported_method")
    with pytest.raises(NotImplementedError, match="Method unsupported_method is not implemented."):
        tiv.compute_pixel_parameter(
            eit_data=mock_eit_data,
            continuous_data=mock_continuous_data,
            sequence=mock_sequence,
        )


@pytest.mark.parametrize(
    ("tiv_method", "expected_error", "expected_result"),
    [
        ("invalid_method", ValueError, None),
        (5, ValueError, None),
        ("inspiratory", None, np.array([8, 8, 8])),
        ("expiratory", None, np.array([8, 10, 8])),
        ("mean", None, np.array([8, 9, 8])),
    ],
)
def test_compute_continuous_parameter_tiv_method(
    mock_continuous_data: MockContinuousData,
    tiv_method: str,
    expected_error: ValueError | None,
    expected_result: np.ndarray,
):
    """Test compute_continuous_parameter with various tiv_method inputs."""
    tiv = TIV()
    if expected_error:
        with pytest.raises(expected_error):
            tiv.compute_continuous_parameter(mock_continuous_data, tiv_method=tiv_method)
    else:
        result = tiv.compute_continuous_parameter(mock_continuous_data, tiv_method=tiv_method)
        assert result.shape == (3,)
        assert np.allclose(result, expected_result, atol=0.01)


@pytest.mark.parametrize(
    ("tiv_method", "expected_error", "expected_result"),
    [
        ("invalid_method", ValueError, None),
        (5, ValueError, None),
        (
            "inspiratory",
            None,
            np.array(
                [
                    [[np.nan, np.nan], [np.nan, np.nan]],
                    [[2, 2], [2, 2]],
                    [[np.nan, np.nan], [np.nan, np.nan]],
                ],
            ),
        ),
        (
            "expiratory",
            None,
            np.array(
                [
                    [[np.nan, np.nan], [np.nan, np.nan]],
                    [[2.5, 2.5], [2.5, 2.5]],
                    [[np.nan, np.nan], [np.nan, np.nan]],
                ],
            ),
        ),
        (
            "mean",
            None,
            np.array(
                [
                    [[np.nan, np.nan], [np.nan, np.nan]],
                    [[2.25, 2.25], [2.25, 2.25]],
                    [[np.nan, np.nan], [np.nan, np.nan]],
                ],
            ),
        ),
    ],
)
def test_compute_pixel_parameter_invalid_tiv_method(
    mock_eit_data: MockEITData,
    mock_continuous_data: MockContinuousData,
    mock_sequence: MockSequence,
    tiv_method: str,
    expected_error: ValueError | None,
    expected_result: np.ndarray,
):
    """Test compute_pixel_parameter with invalid tiv_method."""
    tiv = TIV()
    if expected_error:
        with pytest.raises(expected_error):
            tiv.compute_pixel_parameter(mock_eit_data, mock_continuous_data, mock_sequence, tiv_method=tiv_method)
    else:
        result = tiv.compute_pixel_parameter(
            mock_eit_data,
            mock_continuous_data,
            mock_sequence,
            tiv_method=tiv_method,
        )
        assert result.shape == (3, 2, 2)
        assert np.allclose(
            result[np.isfinite(result)],
            expected_result[np.isfinite(expected_result)],
            atol=0.01,
        )  # isfinite because first and last breaths are expected to be np.nan


@pytest.mark.parametrize(
    ("tiv_timing", "expected_error", "expected_result"),
    [
        ("invalid_timing", ValueError, None),
        (
            "continuous",
            None,
            np.array(
                [
                    [[2, 2], [2, 2]],
                    [[2, 2], [2, 2]],
                    [[2, 2], [2, 2]],
                ],
            ),
        ),
        (
            "pixel",
            None,
            np.array(
                [
                    [[np.nan, np.nan], [np.nan, np.nan]],
                    [[2, 2], [2, 2]],
                    [[np.nan, np.nan], [np.nan, np.nan]],
                ],
            ),
        ),
    ],
)
def test_compute_pixel_parameter_tiv_timing(
    mock_eit_data: MockEITData,
    mock_continuous_data: MockContinuousData,
    mock_sequence: MockSequence,
    tiv_timing: str,
    expected_error: ValueError | None,
    expected_result: np.ndarray,
):
    """Test compute_pixel_parameter with various tiv_timing inputs."""
    tiv = TIV()
    if expected_error and tiv_timing == "invalid_timing":
        with pytest.raises(expected_error, match="tiv_timing must be either 'continuous' or 'pixel'"):
            tiv.compute_pixel_parameter(
                eit_data=mock_eit_data,
                continuous_data=mock_continuous_data,
                sequence=mock_sequence,
                tiv_timing=tiv_timing,
            )
    else:
        result = tiv.compute_pixel_parameter(
            eit_data=mock_eit_data,
            continuous_data=mock_continuous_data,
            sequence=mock_sequence,
            tiv_timing=tiv_timing,
        )
        assert result.shape == (3, 2, 2)
        assert np.allclose(
            result[np.isfinite(result)],
            expected_result[np.isfinite(expected_result)],
            atol=0.01,
        )


def test_tiv_with_no_breaths_continuous(mock_continuous_data: MockContinuousData):
    """Test compute_continuous_parameter when no breaths are detected."""
    tiv = TIV()
    with (
        patch.object(
            tiv,
            "_detect_breaths",
            return_value=IntervalData(
                label="breaths",
                name="No Breaths",
                unit="seconds",
                category="breath",
                intervals=[],
                values=[],
                parameters={},
                derived_from=[],
            ),
        ),
        patch.object(tiv, "_calculate_tiv_values", return_value=[]),
    ):
        result = tiv.compute_continuous_parameter(mock_continuous_data, tiv_method="inspiratory")
        assert result == []


def test_tiv_with_no_breaths_pixel(
    mock_eit_data: MockEITData,
    mock_continuous_data: MockContinuousData,
    mock_sequence: MockSequence,
):
    """Test compute_pixel_parameter when no pixel inflations are detected."""
    tiv = TIV()
    with (
        patch.object(
            tiv,
            "_detect_pixel_inflations",
            return_value=IntervalData(
                label="pixel inflations",
                name="No Pixel Inflations",
                unit="seconds",
                category="breath",
                intervals=[],
                values=np.empty((0, 2, 2), dtype=object),
                parameters={},
                derived_from=[],
            ),
        ),
        patch.object(tiv, "_calculate_tiv_values", return_value=[]),
    ):
        result = tiv.compute_pixel_parameter(
            eit_data=mock_eit_data,
            continuous_data=mock_continuous_data,
            sequence=mock_sequence,
            tiv_method="mean",
            tiv_timing="pixel",
        )
        assert result.shape == (0, 2, 2)


def test_with_data(draeger1: Sequence, timpel1: Sequence, pytestconfig: pytest.Config):
    # Skip test if '--cov' option is enabled
    if pytestconfig.getoption("--cov"):
        pytest.skip("Skip with option '--cov' so other tests can cover 100%.")

    # Make deep copies of the data to avoid modifying the original sequences
    draeger1 = copy.deepcopy(draeger1)
    timpel1 = copy.deepcopy(timpel1)

    # Iterate over both sequences (draeger1 and timpel1)
    for sequence in draeger1, timpel1:
        # Select a subset of the sequence for testing (first 500 samples)
        ssequence = sequence[0:500]

        # Initialize the TIV object
        tiv = TIV()
        eit_data = ssequence.eit_data["raw"]
        cd = ssequence.continuous_data["global_impedance_(raw)"]

        result_continuous = tiv.compute_continuous_parameter(cd, tiv_method="inspiratory")
        result_pixel = tiv.compute_pixel_parameter(eit_data, cd, ssequence)

        assert result_continuous is not None
        assert isinstance(result_continuous, np.ndarray)
        assert result_continuous.ndim == 1
        assert np.all(result_continuous > 0)  # values should be positive for continuous data

        assert result_pixel is not None
        assert isinstance(result_continuous, np.ndarray)
        assert result_pixel.ndim == 3
