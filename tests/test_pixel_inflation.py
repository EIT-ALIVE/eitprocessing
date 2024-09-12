import math
from unittest.mock import patch

import numpy as np
import pytest

from eitprocessing.datahandling.breath import Breath
from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.pixel_inflation import PixelInflation


@pytest.fixture
def mock_pixel_impedance():
    """Mock pixel_impedance with phase shifted cosines for testing."""
    # Create a time vector (e.g., 100 points from 0 to 2*pi)
    time = np.linspace(0, 2 * np.pi, 100)

    # Create 4 cosine waves with different frequencies or phases
    cos_wave_1 = np.cos(time)  # Standard cosine wave
    cos_wave_2 = np.cos(time + np.pi)  # Phase-shifted by pi
    cos_wave_3 = np.cos(time + 0.5 * np.pi)  # Phase-shifted by 0.5*pi
    cos_wave_4 = np.cos(time + 0.25 * np.pi)  # Phase-shifted by 0.25*pi

    # Store the waves in a (time, 2, 2) array
    pixel_impedance = np.empty((len(time), 2, 2))
    pixel_impedance[:, 0, 0] = cos_wave_1
    pixel_impedance[:, 0, 1] = cos_wave_2
    pixel_impedance[:, 1, 0] = cos_wave_3
    pixel_impedance[:, 1, 1] = cos_wave_4

    return time, pixel_impedance


class MockEITData(EITData):
    """Class to create Mock EITData for running tests."""

    def __init__(self):
        """Mock pixel_impedance with phase shifted cosines for testing."""
        # Create a time vector (e.g., 400 points from 0 to 2*pi)
        time = np.linspace(0, 2 * np.pi, 400)

        # Create 4 cosine waves with different frequencies or phases
        cos_wave_1 = np.cos(4 * time)  # Standard cosine wave
        cos_wave_2 = np.cos(4 * time + np.pi)  # Phase-shifted by pi
        cos_wave_3 = np.cos(4 * time + 0.5 * np.pi)  # Phase-shifted by 0.5*pi
        cos_wave_4 = np.cos(4 * time + 0.25 * np.pi)  # Phase-shifted by 0.25*pi

        # Store the waves in a (time, 2, 2) array
        pixel_impedance = np.empty((len(time), 2, 2))
        pixel_impedance[:, 0, 0] = cos_wave_1
        pixel_impedance[:, 0, 1] = cos_wave_2
        pixel_impedance[:, 1, 0] = cos_wave_3
        pixel_impedance[:, 1, 1] = cos_wave_4

        self.time = time
        self.pixel_impedance = pixel_impedance
        self.label = "raw"


class MockContinuousData(ContinuousData):
    """Class to create Mock ContinuousData for running tests."""

    def __init__(self, mock_eit_data: MockEITData):
        pixel_impedance = mock_eit_data.pixel_impedance
        self.time = np.linspace(0, 2 * np.pi, 400)
        self.values = np.nansum(
            pixel_impedance,
            axis=(1, 2),
        )
        self.label = "global_impedance(raw)"
        self.sample_frequency = 399 / 2 * np.pi


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


@pytest.fixture
def mock_continuous_data(mock_eit_data: MockEITData):
    """Fixture to provide an instance of MockContinuousData."""
    return MockContinuousData(mock_eit_data)


@pytest.fixture
def mock_eit_data():
    """Fixture to provide an instance of MockEITData."""
    return MockEITData()


@pytest.fixture
def mock_sequence(mock_eit_data: MockEITData, mock_continuous_data: MockEITData):
    """Fixture to provide an instance of MockSequence."""
    return MockSequence(mock_eit_data, mock_continuous_data)


def test__compute_inflations():
    """Test _compute_inflations helper function."""
    time = np.array([0, 1, 2, 3, 4])
    start = [0, 1]
    middle = [1, 2]
    end = [2, 3]
    pi = PixelInflation()
    result = pi._compute_inflations(start, middle, end, time)

    assert len(result) == 4  # Two inflations plus two None
    assert result[0] is None
    assert result[-1] is None
    assert isinstance(result[1], Breath)


def test__find_extreme_indices(mock_pixel_impedance: tuple):
    """Test _find_extreme_indices helper function."""
    _, pixel_impedance = mock_pixel_impedance
    # Define the time indices where we want to find the extrema
    times = np.array([0, 50])  # Indices between which to find extreme indices

    # Expected min/max indices for each wave
    expected_min_max = [
        (-1, 1),  # cos_wave_1
        (-1, 1),  # cos_wave_2
        (-1, 0),  # cos_wave_3
        (-1, np.sqrt(2) / 2),  # cos_wave_4
    ]

    # Create an instance of PixelInflation
    pi = PixelInflation()

    # Loop over each wave, defined by the (row, col) position in pixel_impedance
    for i, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        # Call the _find_extreme_indices method for min and max
        result_min = pi._find_extreme_indices(pixel_impedance, times, row, col, np.argmin)
        result_max = pi._find_extreme_indices(pixel_impedance, times, row, col, np.argmax)

        # Get the expected min and max for the current wave
        expected_min, expected_max = expected_min_max[i]

        assert math.isclose(pixel_impedance[result_min[0], row, col], expected_min, abs_tol=0.01)
        assert math.isclose(pixel_impedance[result_max[0], row, col], expected_max, abs_tol=0.01)


@pytest.mark.parametrize(
    ("store_input", "sequence_fixture", "expect_error", "expected_exception"),
    [
        (True, "mock_sequence", False, None),  # No error expected
        (True, "not_a_sequence", True, ValueError),  # Expect ValueError because mock_eit_data is not a Sequence
        (True, None, True, RuntimeError),  # Expect RuntimeError because store=True but no Sequence is provided
        (False, "mock_sequence", False, None),  # No error expected, but no result should be stored
        (False, None, False, None),  # No error expected, no sequence provided
        (None, "mock_sequence", False, None),  # No error expected
        (None, None, False, None),  # No error expected
    ],
)
def test_store_result(
    mock_eit_data: MockEITData,
    mock_continuous_data: MockContinuousData,
    request: pytest.FixtureRequest,
    store_input: bool,
    sequence_fixture: str,
    expect_error: bool,
    expected_exception: ValueError | RuntimeError,
):
    """Test storing results in the sequence."""
    pi = PixelInflation(breath_detection_kwargs={"minimum_duration": 0.01})  # ensure that breaths are detected

    # Get the sequence fixture dynamically
    if sequence_fixture == "not_a_sequence":
        sequence = []
    else:
        sequence = request.getfixturevalue(sequence_fixture) if sequence_fixture is not None else None

    if expect_error:
        # Expect a specific exception (either ValueError or RuntimeError)
        with pytest.raises(expected_exception):
            pi.find_pixel_inflations(mock_eit_data, mock_continuous_data, sequence, store=store_input)
    else:
        # Run pixel inflation detection and check the result
        result = pi.find_pixel_inflations(mock_eit_data, mock_continuous_data, sequence, store=store_input)

        # If store is True or None and sequence is not None, check that the result is stored in the sequence
        if (store_input is True or store_input is None) and sequence is not None:
            assert len(sequence.interval_data.data) == 1
            assert sequence.interval_data.data[0] == result
        elif sequence is not None:
            assert len(sequence.interval_data.data) == 0


def mock_compute_pixel_parameter(mean: int):
    def _mock(*_args, **_kwargs) -> np.ndarray:
        if mean < 0:
            return np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]])
        if mean > 0:
            return np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        return np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])

    return _mock


@pytest.mark.parametrize(
    ("mean"),
    [
        0,
        -1,
        1,
    ],
)
def test_with_custom_mean_pixel_tiv(mock_eit_data: MockEITData, mock_continuous_data: MockContinuousData, mean: int):
    mock_function = mock_compute_pixel_parameter(mean)
    with patch(
        "eitprocessing.parameters.tidal_impedance_variation.TIV.compute_pixel_parameter",
        side_effect=mock_function,
    ):
        pi = PixelInflation(breath_detection_kwargs={"minimum_duration": 0.01})

        result = pi.find_pixel_inflations(mock_eit_data, mock_continuous_data)

        assert result.values.shape == (3, 2, 2)

        if mean == 0:
            pass
        else:
            for row in range(2):
                for col in range(2):
                    if mean == 0:
                        # Expect None values when mean == 0
                        assert result.values[1, row, col] is None
                    else:
                        time_point = result.values[1, row, col].middle_time
                        index = np.where(mock_eit_data.time == time_point)[0]
                        value_at_time = mock_eit_data.pixel_impedance[index[0], row, col]
                        if mean == -1:
                            assert math.isclose(value_at_time, -1, abs_tol=0.01)
                        elif mean == 1:
                            assert math.isclose(value_at_time, 1, abs_tol=0.01)
