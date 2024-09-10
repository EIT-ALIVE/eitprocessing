import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from eitprocessing.datahandling.breath import Breath
from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.intervaldata import IntervalData
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


@pytest.fixture
def mock_global_impedance():
    """Mock global_impedance for testing."""
    pixel_impedance = mock_pixel_impedance()
    return np.nansum(pixel_impedance, axis=(1, 2))


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


def test__find_extreme_indices(mock_pixel_impedance):
    """Test _find_extreme_indices helper function."""
    time, pixel_impedance = mock_pixel_impedance
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
    ("mean_value", "expected_mode"),
    [
        (0.0, (np.argmin, np.argmax)),
        (-1.0, (np.argmax, np.argmin)),
        (1.0, (np.argmin, np.argmax)),
    ],
)
def test_pixel_inflation_modes(mean_value, expected_mode):
    """Test mode selection logic based on mean value."""
    pi = PixelInflation()

    # Mock the EITData and ContinuousData objects as needed
    mock_eit_data = MagicMock()
    mock_continuous_data = MagicMock()

    pi.find_pixel_inflations(mock_eit_data, mock_continuous_data)
