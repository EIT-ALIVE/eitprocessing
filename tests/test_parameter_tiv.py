import copy
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.datahandling.sparsedata import SparseData
from eitprocessing.parameters.tidal_impedance_variation import TIV
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


def create_result_array(value: float):
    nan_row = [[np.nan] * 2] * 2
    value_row = [[value] * 2] * 2
    return np.array([nan_row, value_row, nan_row])


def mock_pixel_impedance():
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
        description="Global impedance created for testing TIV parameter",
        parameters={},
        derived_from="mock_eit_data",
        time=np.linspace(0, 18, (18 * 1000), endpoint=False),
        values=mock_global_impedance(),
        sample_frequency=1000,
    )


@pytest.fixture
def mock_eit_data():
    """Fixture to provide an instance of EITData."""
    return EITData(
        path="",
        nframes=2000,
        time=np.linspace(0, 18, (18 * 1000), endpoint=False),
        sample_frequency=1000,
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
        description="Sequence created for parameter TIV testing",
        eit_data=data_collection_eit,
        continuous_data=data_collection_continuous,
        sparse_data=data_collection_sparse,
        interval_data=data_collection_interval,
    )


@pytest.fixture
def not_a_sequence():
    return []


@pytest.fixture
def none_sequence():
    return None


def test_tiv_initialization():
    """Test that TIV initializes correctly with default parameters."""
    tiv = TIV()
    assert tiv.method == "extremes"
    assert tiv.breath_detection == BreathDetection()


def test_deprecated():
    with pytest.warns(DeprecationWarning):
        _ = TIV(breath_detection_kwargs={})

    with pytest.raises(TypeError):
        _ = TIV(breath_detection=BreathDetection(), breath_detection_kwargs={})

    bd_kwargs = {"minimum_duration": 10, "averaging_window_duration": 100.0}

    with pytest.warns(DeprecationWarning):
        assert TIV(breath_detection_kwargs=bd_kwargs).breath_detection == BreathDetection(**bd_kwargs)


def test_compute_parameter_type_error():
    """Test that compute_parameter raises TypeError for unsupported data types."""
    tiv = TIV()

    with pytest.raises(TypeError, match="This method is implemented for ContinuousData or EITData"):
        # Pass an unsupported type to trigger TypeError
        tiv.compute_parameter("unsupported_type")


def test_tiv_initialization_with_invalid_method():
    """Test that TIV raises NotImplementedError for unsupported methods during initialization."""
    with pytest.raises(
        NotImplementedError,
        match="Method unsupported_method is not implemented. The method must be 'extremes'.",
    ):
        TIV(method="unsupported_method")


def test_tiv_initialization_with_valid_method():
    """Test that TIV initializes successfully with a valid method."""
    tiv = TIV(method="extremes")  # This should not raise an error
    assert tiv.method == "extremes"


@pytest.mark.parametrize(
    ("store_input", "sequence_fixture", "expected_exception"),
    [
        (True, "not_a_sequence", ValueError),  # Expect ValueError because a string is not a valid Sequence
        (True, "none_sequence", RuntimeError),  # Expect RuntimeError because sequence is None and store=True
    ],
)
def test_store_result_with_errors(
    mock_eit_data: EITData,
    mock_continuous_data: ContinuousData,
    request: pytest.FixtureRequest,
    store_input: bool,
    sequence_fixture: str,
    expected_exception: ValueError | RuntimeError,
):
    """Test storing results when errors are expected."""
    tiv = TIV()  # Ensure that breaths are detected

    # Retrieve the sequence from the fixture
    test_sequence = request.getfixturevalue(sequence_fixture)

    # Expect a specific exception (either ValueError or RuntimeError)
    with pytest.raises(expected_exception):
        tiv.compute_continuous_parameter(
            mock_continuous_data,
            tiv_method="inspiratory",
            sequence=test_sequence,
            store=store_input,
        )
    with pytest.raises(expected_exception):
        tiv.compute_pixel_parameter(
            eit_data=mock_eit_data,
            continuous_data=mock_continuous_data,
            sequence=test_sequence,
            tiv_method="inspiratory",
            tiv_timing="pixel",
            store=store_input,
            result_label="pixel_tivs",
        )


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
    tiv = TIV()  # Ensure that breaths are detected

    # Retrieve the sequence from the fixture
    test_sequence = request.getfixturevalue(sequence_fixture)

    # Run continuous and pixel tiv computation and check the result
    continuous_result = tiv.compute_continuous_parameter(
        mock_continuous_data,
        tiv_method="inspiratory",
        sequence=test_sequence,
        store=store_input,
        result_label="continuous_tivs",
    )

    pixel_result = tiv.compute_pixel_parameter(
        eit_data=mock_eit_data,
        continuous_data=mock_continuous_data,
        sequence=test_sequence,
        tiv_method="inspiratory",
        tiv_timing="pixel",
        store=store_input,
        result_label="pixel_tivs",
    )

    # Check that the results are stored correctly based on store_input
    if store_input in [True, None] and test_sequence is not None:
        assert len(test_sequence.sparse_data.data) == 2
        assert test_sequence.sparse_data["continuous_tivs"] == continuous_result
        assert test_sequence.sparse_data["pixel_tivs"] == pixel_result
    elif test_sequence is not None:
        assert len(test_sequence.sparse_data.data) == 0


@pytest.mark.parametrize(
    ("tiv_method", "expected_error"),
    [
        ("invalid_method", ValueError),
        (5, ValueError),
    ],
)
def test_compute_continuous_parameter_tiv_method_errors(
    mock_continuous_data: ContinuousData,
    tiv_method: str,
    expected_error: ValueError,
):
    """Test compute_continuous_parameter with invalid tiv_method inputs that raise errors."""
    tiv = TIV()
    with pytest.raises(expected_error):
        tiv.compute_continuous_parameter(mock_continuous_data, tiv_method=tiv_method)


@pytest.mark.parametrize(
    ("tiv_method", "expected_result"),
    [
        ("inspiratory", np.array([8, 8, 8])),
        ("expiratory", np.array([8, 10, 8])),
        ("mean", np.array([8, 9, 8])),
    ],
)
def test_compute_continuous_parameter_tiv_method_success(
    mock_continuous_data: ContinuousData,
    tiv_method: str,
    expected_result: np.ndarray,
):
    """Test compute_continuous_parameter with valid tiv_method inputs that return expected results."""
    tiv = TIV()
    result = tiv.compute_continuous_parameter(mock_continuous_data, tiv_method=tiv_method)
    result = np.stack(result.values)
    assert result.shape == (3,)
    assert np.allclose(result, expected_result, atol=0.01)


@pytest.mark.parametrize(
    ("tiv_method", "expected_error"),
    [
        ("invalid_method", ValueError),
        (5, ValueError),
    ],
)
def test_compute_pixel_parameter_invalid_tiv_method_errors(
    mock_eit_data: EITData,
    mock_continuous_data: ContinuousData,
    mock_sequence: Sequence,
    tiv_method: str,
    expected_error: ValueError,
):
    """Test compute_pixel_parameter with invalid tiv_method inputs that raise errors."""
    tiv = TIV()
    with pytest.raises(expected_error):
        tiv.compute_pixel_parameter(mock_eit_data, mock_continuous_data, mock_sequence, tiv_method=tiv_method)


@pytest.mark.parametrize(
    ("tiv_method", "expected_result"),
    [
        ("inspiratory", create_result_array(2)),
        ("expiratory", create_result_array(2.5)),
        ("mean", create_result_array(2.25)),
    ],
)
def test_compute_pixel_parameter_valid_tiv_method(
    mock_eit_data: EITData,
    mock_continuous_data: ContinuousData,
    mock_sequence: Sequence,
    tiv_method: str,
    expected_result: np.ndarray,
):
    """Test compute_pixel_parameter with valid tiv_method inputs that return expected results."""
    tiv = TIV()
    result = tiv.compute_pixel_parameter(
        mock_eit_data,
        mock_continuous_data,
        mock_sequence,
        tiv_method=tiv_method,
    )
    result = np.stack(result.values)
    assert result.shape == (3, 2, 2)
    assert np.allclose(
        result[np.isfinite(result)],
        expected_result[np.isfinite(expected_result)],
        atol=0.01,
    )  # isfinite because first and last breaths are expected to be np.nan


@pytest.mark.parametrize(
    ("tiv_timing", "expected_error"),
    [
        ("invalid_timing", ValueError),
    ],
)
def test_compute_pixel_parameter_tiv_timing_errors(
    mock_eit_data: EITData,
    mock_continuous_data: ContinuousData,
    mock_sequence: Sequence,
    tiv_timing: str,
    expected_error: ValueError,
):
    """Test compute_pixel_parameter with invalid tiv_timing inputs that raise errors."""
    tiv = TIV()
    with pytest.raises(expected_error, match="tiv_timing must be either 'continuous' or 'pixel'"):
        tiv.compute_pixel_parameter(
            eit_data=mock_eit_data,
            continuous_data=mock_continuous_data,
            sequence=mock_sequence,
            tiv_timing=tiv_timing,
        )


@pytest.mark.parametrize(
    ("tiv_timing", "expected_result"),
    [
        ("continuous", np.full((3, 2, 2), 2)),
        ("pixel", create_result_array(2)),
    ],
)
def test_compute_pixel_parameter_tiv_timing_success(
    mock_eit_data: EITData,
    mock_continuous_data: ContinuousData,
    mock_sequence: Sequence,
    tiv_timing: str,
    expected_result: np.ndarray,
):
    """Test compute_pixel_parameter with valid tiv_timing inputs that return expected results."""
    tiv = TIV()
    result = tiv.compute_pixel_parameter(
        eit_data=mock_eit_data,
        continuous_data=mock_continuous_data,
        sequence=mock_sequence,
        tiv_timing=tiv_timing,
    )
    result = np.stack(result.values)
    assert result.shape == (3, 2, 2)
    assert np.allclose(
        result[np.isfinite(result)],
        expected_result[np.isfinite(expected_result)],
        atol=0.01,
    )


def test_tiv_with_no_breaths_continuous(mock_continuous_data: ContinuousData):
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
        assert len(result.values) == 0


def test_tiv_with_no_breaths_pixel(
    mock_eit_data: EITData,
    mock_continuous_data: ContinuousData,
    mock_sequence: Sequence,
):
    """Test compute_pixel_parameter when no pixel breaths are detected."""
    tiv = TIV()
    with (
        patch.object(
            tiv,
            "_detect_pixel_breaths",
            return_value=IntervalData(
                label="pixel breaths",
                name="No Pixel breaths",
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
        result = np.empty((0, 2, 2)) if not len(result.values) else np.stack(result.values)
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
        # Initialize the TIV object
        tiv = TIV()
        eit_data = sequence.eit_data["raw"]
        cd = sequence.continuous_data["global_impedance_(raw)"]

        result_continuous = tiv.compute_continuous_parameter(cd, tiv_method="inspiratory")
        result_pixel = tiv.compute_pixel_parameter(eit_data, cd, sequence)

        arr_result_continuous = np.stack(result_continuous.values)
        arr_result_pixel = np.stack(result_pixel.values)

        assert result_continuous is not None
        assert isinstance(result_continuous, SparseData)
        assert arr_result_continuous.ndim == 1
        assert np.all(arr_result_continuous > 0)  # values should be positive for continuous data

        assert result_pixel is not None
        assert isinstance(result_pixel, SparseData)
        assert arr_result_pixel.ndim == 3


@pytest.mark.parametrize(
    ("bd_kwargs", "expected_error"),
    [
        ({"amplitude_cutoff_fraction": 0.3, "minimum_duration": 5.0}, ValueError),  # too long duration
        ({"amplitude_cutoff_fraction": 2, "minimum_duration": 5.0}, ValueError),  # too high amplitude cutoff
    ],
)
def test_detect_pixel_breaths_with_invalid_bd_kwargs(
    bd_kwargs: dict,
    expected_error: type[Exception],
    mock_eit_data: EITData,
    mock_continuous_data: ContinuousData,
    mock_sequence: Sequence,
):
    """Test detect_pixel_breaths with invalid bd_kwargs that raise errors."""
    tiv = TIV(breath_detection=BreathDetection(**bd_kwargs))

    with pytest.raises(expected_error):
        tiv._detect_pixel_breaths(mock_eit_data, mock_continuous_data, mock_sequence, store=False)


@pytest.mark.parametrize(
    ("bd_kwargs", "expected_error"),
    [
        ({"minimum_amplitude": 2, "minimum_duration": 5.0}, TypeError),  # unexpected keyword minimum_amplitude
    ],
)
def test_detect_pixel_breaths_with_invalid_bd_kwargs_(
    bd_kwargs: dict,
    expected_error: type[Exception],
):
    """Test detect_pixel_breaths with invalid bd_kwargs that raise errors."""
    with pytest.raises(expected_error):
        _ = TIV(breath_detection=BreathDetection(**bd_kwargs))


@pytest.mark.parametrize(
    ("bd_kwargs"),
    [
        ({"amplitude_cutoff_fraction": 0.1, "minimum_duration": 0.5}),
        ({"amplitude_cutoff_fraction": 0.2, "minimum_duration": 0.3}),
        ({"amplitude_cutoff_fraction": 0.5, "minimum_duration": 0.2}),
    ],
)
def test_detect_pixel_breaths_with_valid_bd_kwargs(
    bd_kwargs: dict,
    mock_eit_data: EITData,
    mock_continuous_data: ContinuousData,
    mock_sequence: Sequence,
):
    """Test detect_pixel_breaths with valid bd_kwargs that return expected results."""
    tiv = TIV(breath_detection=BreathDetection(**bd_kwargs))

    result = tiv._detect_pixel_breaths(mock_eit_data, mock_continuous_data, mock_sequence, store=False)
    test_result = np.stack(result.values)
    # Assert that the result is of the expected type and shape
    assert isinstance(result, IntervalData)
    assert test_result.shape == (3, 2, 2)  # Adjust this based on your expected output
