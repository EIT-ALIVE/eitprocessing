import copy
import itertools
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from eitprocessing.datahandling.breath import Breath
from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.datahandling.sparsedata import SparseData
from eitprocessing.features.pixel_breath import PixelBreath
from tests.test_breath_detection import BreathDetection

environment = Path(
    os.environ.get(
        "EIT_PROCESSING_TEST_DATA",
        Path(__file__).parent.parent.resolve(),
    ),
)
data_directory = environment / "tests" / "test_data"
draeger_file1 = data_directory / "Draeger_Test3.bin"
timpel_file = data_directory / "Timpel_Test.txt"


# @pytest.fixture()
def mock_pixel_impedance():
    """Mock pixel_impedance with phase shifted cosines for testing."""
    # Create a time vector
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

    return pixel_impedance


def mock_pixel_impedance_one_zero() -> np.ndarray:
    pixel_impedance = mock_pixel_impedance()
    pixel_impedance[:, 1, 1] = np.abs(pixel_impedance[:, 1, 1] * 0)
    return pixel_impedance


def mock_global_impedance():
    pixel_impedance = mock_pixel_impedance()
    return np.nansum(
        pixel_impedance,
        axis=(1, 2),
    )


@pytest.fixture
def mock_continuous_data():
    """Fixture to provide an instance of ContinuousData."""
    return ContinuousData(
        label="global_impedance",
        name="global_impedance",
        unit="au",
        category="relative impedance",
        description="Global impedance created for testing pixel breath feature",
        parameters={},
        derived_from="mock_eit_data",
        time=np.linspace(0, 2 * np.pi, 400),
        values=mock_global_impedance(),
        sample_frequency=399 / 2 * np.pi,
    )


@pytest.fixture
def mock_eit_data():
    """Fixture to provide an instance of EITData."""
    return EITData(
        path="",
        nframes=400,
        time=np.linspace(0, 2 * np.pi, 400),
        sample_frequency=399 / 2 * np.pi,
        vendor=Vendor.DRAEGER,
        label="mock_eit_data",
        name="mock_eit_data",
        pixel_impedance=mock_pixel_impedance(),
    )


@pytest.fixture
def mock_sequence(mock_eit_data: EITData, mock_continuous_data: ContinuousData):
    """Fixture to provide an instance of Sequence."""
    data_collection_eit = DataCollection(EITData)
    data_collection_eit.add(mock_eit_data)

    data_collection_continuous = DataCollection(ContinuousData)
    data_collection_continuous.add(mock_continuous_data)

    data_collection_sparse = DataCollection(SparseData)
    data_collection_interval = DataCollection(IntervalData)

    return Sequence(
        label="mock_sequence",
        name="mock_sequence",
        description="Sequence created for pixel breath feature testing",
        eit_data=data_collection_eit,
        continuous_data=data_collection_continuous,
        sparse_data=data_collection_sparse,
        interval_data=data_collection_interval,
    )


@pytest.fixture
def mock_zero_eit_data():
    """Fixture to provide an instance of EITData with one element set to zero."""
    return EITData(
        path="",
        nframes=400,
        time=np.linspace(0, 2 * np.pi, 400),
        sample_frequency=399 / 2 * np.pi,
        vendor=Vendor.DRAEGER,
        label="mock_eit_data",
        name="mock_eit_data",
        pixel_impedance=mock_pixel_impedance_one_zero(),
    )


@pytest.fixture
def mock_only_pixel_impedance():
    """Mock pixel_impedance with phase shifted cosines for testing."""
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
def not_a_sequence():
    return []


@pytest.fixture
def none_sequence():
    return None


def mock_compute_pixel_parameter(mean: int):
    def _mock(*_args, **_kwargs) -> SparseData:
        return SparseData(
            label="mock_sparse_data",
            name="Tidal impedance variation",
            unit=None,
            category="impedance difference",
            time=np.linspace(0, 100, 100),
            description="Mock tidal impedance variation",
            parameters={},
            derived_from=[],
            values=np.full(100, np.sign(mean)),
        )

    return _mock


def test_deprecated():
    with pytest.warns(DeprecationWarning):
        _ = PixelBreath(breath_detection_kwargs={})

    with pytest.raises(TypeError):
        _ = PixelBreath(breath_detection=BreathDetection(), breath_detection_kwargs={})

    bd_kwargs = {"minimum_duration": 10, "averaging_window_duration": 100.0}
    with pytest.warns(DeprecationWarning):
        assert PixelBreath(breath_detection_kwargs=bd_kwargs).breath_detection == BreathDetection(**bd_kwargs)


def test__compute_breaths():
    """Test _compute_breaths helper function."""
    time = np.array([0, 1, 2, 3, 4])
    start = [0, 2]
    middle = [1, 3]
    end = [2, 4]
    pi = PixelBreath()
    result = pi._construct_breaths(start, middle, end, time)

    assert len(result) == 4  # Two breaths plus two None
    assert result[0] is None
    assert result[-1] is None
    assert isinstance(result[1], Breath)


def test__find_extreme_indices(mock_only_pixel_impedance: tuple):
    """Test _find_extreme_indices helper function."""
    _, pixel_impedance = mock_only_pixel_impedance
    # Define the time indices where we want to find the extrema
    indices = np.array([0, 50])  # Indices between which to find extreme indices

    # Expected min/max indices for each wave
    expected_min_max = [
        (-1, 1),  # cos_wave_1
        (-1, 1),  # cos_wave_2
        (-1, 0),  # cos_wave_3
        (-1, np.sqrt(2) / 2),  # cos_wave_4
    ]

    # Create an instance of PixelBreath
    pi = PixelBreath()

    # Loop over each wave, defined by the (row, col) position in pixel_impedance
    for i, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        # Call the _find_extreme_indices method for min and max
        result_min = pi._find_extreme_indices(pixel_impedance, indices, row, col, np.argmin)
        result_max = pi._find_extreme_indices(pixel_impedance, indices, row, col, np.argmax)

        # Get the expected min and max for the current wave
        expected_min, expected_max = expected_min_max[i]

        assert np.isclose(pixel_impedance[result_min[0], row, col], expected_min, atol=0.01)
        assert np.isclose(pixel_impedance[result_max[0], row, col], expected_max, atol=0.01)


@pytest.mark.parametrize(
    ("store_input", "sequence_fixture", "expected_exception"),
    [
        (True, "not_a_sequence", ValueError),  # Expect ValueError because an empty list is not a Sequence
        (True, "none_sequence", RuntimeError),  # Expect RuntimeError because store=True but no Sequence is provided
    ],
)
def test_store_result_with_errors(
    mock_eit_data: EITData,
    mock_continuous_data: ContinuousData,
    request: pytest.FixtureRequest,
    store_input: bool,
    sequence_fixture: str,
    expected_exception: type[ValueError | RuntimeError],
):
    """Test storing results when errors are expected."""
    pi = PixelBreath(breath_detection=BreathDetection(minimum_duration=0.01))  # Ensure that breaths are detected

    sequence = request.getfixturevalue(sequence_fixture)

    # Expect a specific exception (either ValueError or RuntimeError)
    with pytest.raises(expected_exception):
        pi.find_pixel_breaths(mock_eit_data, mock_continuous_data, sequence, store=store_input)


@pytest.mark.parametrize(
    ("store_input", "sequence_fixture"),
    [
        (True, "mock_sequence"),  # Result should be stored
        (False, "mock_sequence"),  # No result should be stored
        (False, "none_sequence"),  # No result should be stored, no sequence provided
        (None, "mock_sequence"),  # Result should be stored
        (None, "none_sequence"),  # No result stored, no sequence provided
    ],
)
def test_store_result_success(
    mock_eit_data: EITData,
    mock_continuous_data: ContinuousData,
    request: pytest.FixtureRequest,
    store_input: bool,
    sequence_fixture: str,
):
    """Test storing results when no errors are expected."""
    pi = PixelBreath(breath_detection=BreathDetection(minimum_duration=0.01))  # Ensure that breaths are detected

    sequence = request.getfixturevalue(sequence_fixture)

    # Run pixel breath detection and check the result
    result = pi.find_pixel_breaths(mock_eit_data, mock_continuous_data, sequence, store=store_input)

    # If store is True or None and sequence is not None, check that the result is stored in the sequence
    if (store_input is True or store_input is None) and sequence is not None:
        assert len(sequence.interval_data.data) == 1
        assert sequence.interval_data["pixel_breaths"] == result
    elif sequence is not None:
        assert len(sequence.interval_data.data) == 0


@pytest.mark.parametrize(
    ("mean"),
    [
        -1,
        1,
    ],
)
def test_with_custom_mean_pixel_tiv(
    mock_eit_data: EITData,
    mock_continuous_data: ContinuousData,
    mean: int,
):
    mock_function = mock_compute_pixel_parameter(mean)
    with patch(
        "eitprocessing.parameters.tidal_impedance_variation.TIV.compute_pixel_parameter",
        side_effect=mock_function,
    ):
        pi = PixelBreath(breath_detection=BreathDetection(minimum_duration=0.01))

        result = pi.find_pixel_breaths(mock_eit_data, mock_continuous_data)

        test_result = np.stack(result.values)
        assert test_result.shape == (3, 2, 2)

        for row, col in itertools.product(range(2), range(2)):
            time_point = test_result[1, row, col].middle_time
            index = np.where(mock_eit_data.time == time_point)[0]
            value_at_time = mock_eit_data.pixel_impedance[index[0], row, col]
            if mean == -1:
                assert np.isclose(value_at_time, -1, atol=0.01)
            elif mean == 1:
                assert np.isclose(value_at_time, 1, atol=0.01)


def test_with_zero_impedance(mock_zero_eit_data: EITData, mock_continuous_data: ContinuousData):
    pi = PixelBreath(breath_detection=BreathDetection(minimum_duration=0.01))
    breath_container = pi.find_pixel_breaths(mock_zero_eit_data, mock_continuous_data)
    test_result = np.stack(breath_container.values)
    assert np.all(np.vectorize(lambda x: x is None)(test_result[:, 1, 1]))
    assert test_result.shape == (3, 2, 2)


def test_with_data(draeger1: Sequence, timpel1: Sequence, pytestconfig: pytest.Config):
    if pytestconfig.getoption("--cov"):
        pytest.skip("Skip with option '--cov' so other tests can cover 100%.")

    draeger1 = copy.deepcopy(draeger1)
    timpel1 = copy.deepcopy(timpel1)
    for sequence in draeger1, timpel1:
        ssequence = sequence
        pi = PixelBreath()
        eit_data = ssequence.eit_data["raw"]
        cd = ssequence.continuous_data["global_impedance_(raw)"]
        pixel_breaths = pi.find_pixel_breaths(eit_data, cd)
        test_result = np.stack(pixel_breaths.values)
        assert not np.all(test_result == None)  # noqa: E711
        _, n_rows, n_cols = test_result.shape

        for row, col in itertools.product(range(n_rows), range(n_cols)):
            filtered_values = [val for val in test_result[:, row, col] if val is not None]
            if not filtered_values:
                return
            start_indices, middle_indices, end_indices = (list(x) for x in zip(*filtered_values, strict=True))
            # Test whether pixel breaths are sorted properly
            assert start_indices == sorted(start_indices)
            assert middle_indices == sorted(middle_indices)
            assert end_indices == sorted(end_indices)

            # Test whether indices are unique. `set` removes non-unique values,
            # `sorted(list(...))` converts the set to a sorted list again.
            assert list(start_indices) == sorted(set(start_indices))
            assert list(middle_indices) == sorted(set(middle_indices))
            assert list(end_indices) == sorted(set(end_indices))

            # Test whether the start of the next breath is on/after the previous breath
            assert all(
                start_index >= end_index
                for start_index, end_index in zip(start_indices[1:], end_indices[:-1], strict=True)
            )
            for breath in filtered_values:
                # Test whether the indices are in the proper order within a breath
                assert breath.start_time < breath.middle_time < breath.end_time


def test_phase_modes(draeger1: Sequence, pytestconfig: pytest.Config):
    if pytestconfig.getoption("--cov"):
        pytest.skip("Skip with option '--cov' so other tests can cover 100%.")

    ssequence = draeger1
    eit_data = ssequence.eit_data["raw"]

    # reduce the pixel set to middly 'well-behaved' pixels with positive TIV
    eit_data.pixel_impedance = eit_data.pixel_impedance[:, 10:23, 10:23]

    # flip a single pixel, so the differences between algorithms becomes predictable
    eit_data.pixel_impedance[:, 6, 6] = -eit_data.pixel_impedance[:, 6, 6]

    cd = ssequence.continuous_data["global_impedance_(raw)"]

    # replace the 'global' data with the sum of the middly pixels
    cd.values = np.sum(eit_data.pixel_impedance, axis=(1, 2))

    pb_negative_amplitude = PixelBreath(phase_correction_mode="negative amplitude").find_pixel_breaths(eit_data, cd)
    pb_phase_shift = PixelBreath(phase_correction_mode="phase shift").find_pixel_breaths(eit_data, cd)

    # results are not compared, other than for length; just make sure it runs
    pb_none = PixelBreath(phase_correction_mode="none").find_pixel_breaths(eit_data, cd)

    assert len(pb_negative_amplitude) == len(pb_phase_shift) == len(pb_none)

    # all breaths, except for the first and last,  should have been detected
    assert not np.any(np.array(pb_negative_amplitude.values)[1:-1] == None)  # noqa: E711
    assert not np.any(np.array(pb_phase_shift.values)[1:-1] == None)  # noqa: E711

    same_pixel_timing = np.array(pb_negative_amplitude.values) == np.array(pb_phase_shift.values)
    assert not np.all(same_pixel_timing)
    assert not np.any(same_pixel_timing[1:-1, 6, 6])  # the single flipped pixel
    assert np.all(same_pixel_timing[1:-1, :6, :])  # all pixels in the rows above match
    assert np.all(same_pixel_timing[1:-1, 7:, :])  # all pixels in the rows below match
    assert np.all(same_pixel_timing[1:-1, :, :6])  # all pixels in the columns to the left match
    assert np.all(same_pixel_timing[1:-1, :, 7:])  # all pixels in the columns to the right match
    assert np.all(same_pixel_timing[0, :, :])  # all first values match, because they are all None
    assert np.all(same_pixel_timing[-1, :, :])  # all last values match, because they are all None
