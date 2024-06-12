import copy
import os
from pathlib import Path

import numpy as np
import pytest

from eitprocessing.datahandling.breath import Breath
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.breath_detection import BreathDetection

environment = Path(
    os.environ.get(
        "EIT_PROCESSING_TEST_DATA",
        Path(__file__).parent.parent.resolve(),
    ),
)
data_directory = environment / "tests" / "test_data"
draeger_file1 = data_directory / "Draeger_Test3.bin"
draeger_file2 = data_directory / "Draeger_Test.bin"
timpel_file = data_directory / "Timpel_Test.txt"


def make_cosine_wave(sample_frequency: float, length: int, frequency: float):
    """Generate a cosine wave with the given parameters and amplitude 1.

    A cosine wave starts and ends at a value of 1, passing through -1 halfway between the maximum values. This makes it
    a very predictable signal for breath detection.
    """
    time = np.arange(length) / sample_frequency
    return time, np.cos(time * np.pi * 2 * frequency)


def test_find_features():
    pytest.skip()


test_data_remove_edge_cases = [
    (  # test removal of edge peaks
        np.array([0, 1, 3, 5, 7, 9]),
        np.array([2, 4, 6, 8]),
        np.array([1] * 10),
        np.array([3, 5, 7]),
        np.array([2, 4, 6, 8]),
    ),
    (  # test removal of first valley
        np.array([1, 3, 5, 7, 9]),
        np.array([2, 4, 6, 8]),
        np.array([1, 1, 3, *[1] * 7]),
        np.array([5, 7]),
        np.array([4, 6, 8]),
    ),
    (  # test removal of last valley
        np.array([1, 3, 5, 7, 9]),
        np.array([2, 4, 6, 8]),
        np.array([*[1] * 8, 3, 1]),
        np.array([3, 5]),
        np.array([2, 4, 6]),
    ),
]


@pytest.mark.parametrize(
    ("peak_indices", "valley_indices", "moving_average", "expected_peak_indices", "expected_valley_indices"),
    test_data_remove_edge_cases,
)
def test_remove_edge_cases(
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
    moving_average: np.ndarray,
    expected_peak_indices: np.ndarray,
    expected_valley_indices: np.ndarray,
):
    data = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    bd = BreathDetection(1)
    result_peak_indices, result_valley_indices = bd._remove_edge_cases(  # noqa: SLF001
        data,
        peak_indices,
        valley_indices,
        moving_average,
    )
    assert np.array_equal(result_peak_indices, expected_peak_indices)
    assert np.array_equal(result_valley_indices, expected_valley_indices)


test_data_remove_doubles = [
    (  # test removal of the first valley
        np.array([5, 1, 2, 5]),
        np.array([0, 3]),
        np.array([1, 2]),
        np.array([0, 3]),
        np.array([1]),
    ),
    (  # test removal of the second valley
        np.array([5, 2, 1, 5]),
        np.array([0, 3]),
        np.array([1, 2]),
        np.array([0, 3]),
        np.array([2]),
    ),
    (  # test removal of valleys with same value
        np.array([0, 1, 1, 0]),
        np.array([0, 3]),
        np.array([1, 2]),
        np.array([0, 3]),
        np.array([1]),
    ),
    (  # test removal of first peak
        np.array([0, 1, 2, 0]),
        np.array([1, 2]),
        np.array([0, 3]),
        np.array([2]),
        np.array([0, 3]),
    ),
    (  # test removal of second peak
        np.array([0, 2, 1, 0]),
        np.array([1, 2]),
        np.array([0, 3]),
        np.array([1]),
        np.array([0, 3]),
    ),
    (  # test removal of peaks with same value
        np.array([0, 1, 1, 0]),
        np.array([1, 2]),
        np.array([0, 3]),
        np.array([1]),
        np.array([0, 3]),
    ),
    (  # test removal of both valleys and peaks
        np.array([0, 1, 1, 0, 1, 1, 2, 0]),
        np.array([1, 2, 5, 6]),
        np.array([0, 3, 4, 7]),
        np.array([1, 6]),
        np.array([0, 3, 7]),
    ),
    (
        np.array([]),
        np.array([], dtype=int),
        np.array([], dtype=int),
        np.array([]),
        np.array([]),
    ),
]


@pytest.mark.parametrize(
    (
        "data",
        "peak_indices",
        "valley_indices",
        "expected_peak_indices",
        "expected_valley_indices",
    ),
    test_data_remove_doubles,
)
def test_remove_doubles(
    data: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
    expected_peak_indices: np.ndarray,
    expected_valley_indices: np.ndarray,
):
    bd = BreathDetection(sample_frequency=1)
    result_peak_indices, result_valley_indices = bd._remove_doubles(
        data,
        peak_indices,
        valley_indices,
    )

    assert np.all(result_peak_indices == expected_peak_indices)
    assert np.all(result_valley_indices == expected_valley_indices)


test_data_remove_low_amplitude = [
    (  # test remove none
        np.array([0, 1, 3, 1, 3, 1, 3, 1, 3, 1]),
        np.array([2, 4, 6, 8]),
        np.array([1, 3, 5, 7, 9]),
        np.array([2, 4, 6, 8]),
        np.array([1, 3, 5, 7, 9]),
    ),
    (  # test remove first peak, keeping the first valley
        np.array([0, 1, 1.2, 1.1, 3, 1, 3, 1, 3, 1]),
        np.array([2, 4, 6, 8]),
        np.array([1, 3, 5, 7, 9]),
        np.array([4, 6, 8]),
        np.array([1, 5, 7, 9]),
    ),
    (  # test remove first peak, keeping the second valley
        np.array([0, 1.1, 1.2, 1, 3, 1, 3, 1, 3, 1]),
        np.array([2, 4, 6, 8]),
        np.array([1, 3, 5, 7, 9]),
        np.array([4, 6, 8]),
        np.array([3, 5, 7, 9]),
    ),
    (  # test remove second peak, keeping the second valley
        np.array([0, 1, 3, 1, 1.2, 1.1, 3, 1, 3, 1]),
        np.array([2, 4, 6, 8]),
        np.array([1, 3, 5, 7, 9]),
        np.array([2, 6, 8]),
        np.array([1, 3, 7, 9]),
    ),
    (  # test remove second peak, keeping the third valley
        np.array([0, 1, 3, 1.1, 1.2, 1, 3, 1, 3, 1]),
        np.array([2, 4, 6, 8]),
        np.array([1, 3, 5, 7, 9]),
        np.array([2, 6, 8]),
        np.array([1, 5, 7, 9]),
    ),
    (  # test remove last peak, keeping the second to last valley
        np.array([0, 1, 3, 1, 3, 1, 3, 1, 1.2, 1.1]),
        np.array([2, 4, 6, 8]),
        np.array([1, 3, 5, 7, 9]),
        np.array([2, 4, 6]),
        np.array([1, 3, 5, 7]),
    ),
    (  # test remove last peak, keeping the last valley
        np.array([0, 1, 3, 1, 3, 1, 3, 1.1, 1.2, 1]),
        np.array([2, 4, 6, 8]),
        np.array([1, 3, 5, 7, 9]),
        np.array([2, 4, 6]),
        np.array([1, 3, 5, 9]),
    ),
]


@pytest.mark.parametrize(
    (
        "data",
        "peak_indices",
        "valley_indices",
        "expected_peak_indices",
        "expected_valley_indices",
    ),
    test_data_remove_low_amplitude,
)
def test_remove_low_amplitudes(
    data: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
    expected_peak_indices: np.ndarray,
    expected_valley_indices: np.ndarray,
):
    bd = BreathDetection(1)
    result_peak_indices, result_valley_indices = bd._remove_low_amplitudes(data, peak_indices, valley_indices)  # noqa: SLF001
    assert np.array_equal(result_peak_indices, expected_peak_indices)
    assert np.array_equal(result_valley_indices, expected_valley_indices)


@pytest.mark.parametrize(
    (
        "data",
        "peak_indices",
        "valley_indices",
        "expected_peak_indices",
        "expected_valley_indices",
    ),
    test_data_remove_low_amplitude,
)
def test_no_remove_low_amplitudes(
    data: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
    expected_peak_indices: np.ndarray,
    expected_valley_indices: np.ndarray,
):
    """This test uses the same data as test_remove_low_amplitudes, but expects the output to be the same as the input."""
    bd = BreathDetection(1, amplitude_cutoff_fraction=None)
    result_peak_indices, result_valley_indices = bd._remove_low_amplitudes(data, peak_indices, valley_indices)  # noqa: SLF001
    assert np.array_equal(result_peak_indices, peak_indices)
    assert np.array_equal(result_valley_indices, valley_indices)

    bd = BreathDetection(1, amplitude_cutoff_fraction=0)
    result_peak_indices, result_valley_indices = bd._remove_low_amplitudes(data, peak_indices, valley_indices)  # noqa: SLF001
    assert np.array_equal(result_peak_indices, peak_indices)
    assert np.array_equal(result_valley_indices, valley_indices)


def test_remove_no_breaths_around_valid_data():
    sample_frequency = 20
    time, y = make_cosine_wave(sample_frequency, 1000, 1)
    bd = BreathDetection(sample_frequency)

    peak_indices, valley_indices = bd._detect_peaks_and_valleys(y)
    assert np.array_equal(time[peak_indices], np.arange(1, 50))
    assert np.array_equal(
        time[valley_indices],
        np.arange(0, 50) + 0.5,
    )


def test_remove_breaths_around_invalid_data():
    sample_frequency = 20
    time, y = make_cosine_wave(sample_frequency, 1000, 1)
    bd = BreathDetection(sample_frequency)

    valid_peak_indices, valid_valley_indices = bd._detect_peaks_and_valleys(y)

    start_invalid = np.argmax(time >= 2.75)
    end_invalid = np.argmax(time >= 3.25)
    y_with_invalid_data = np.copy(y)
    y_with_invalid_data[start_invalid:end_invalid] = np.nan
    y_with_invalid_data = bd._twosidedfill(y_with_invalid_data)

    invalid_peak_indices, invalid_valley_indices = bd._detect_peaks_and_valleys(y_with_invalid_data)
    assert not np.array_equal(valid_peak_indices, invalid_peak_indices)
    assert not np.array_equal(valid_valley_indices, invalid_valley_indices)

    assert np.array_equal(time[invalid_peak_indices], np.concatenate([[1.0, 2.0], np.arange(4.0, 50.0, 1.0)]))
    assert np.array_equal(
        time[invalid_valley_indices],
        np.concatenate([[0.5, 1.5, 2.5], np.arange(4.5, 50.0, 1.0)]),
    )


def test_with_data(draeger1: Sequence, draeger2: Sequence, timpel1: Sequence, pytestconfig: pytest.Config):
    if pytestconfig.getoption("--cov"):
        pytest.skip("Skip with option '--cov' so other tests can cover 100%.")

    draeger1 = copy.deepcopy(draeger1)
    draeger2 = copy.deepcopy(draeger2)
    timpel1 = copy.deepcopy(timpel1)
    for sequence in draeger1, draeger2, timpel1:
        bd = BreathDetection(
            sample_frequency=sequence.eit_data["raw"].framerate,
        )

        cd = sequence.continuous_data["global_impedance_(raw)"]
        breaths = bd.find_breaths(sequence, "global_impedance_(raw)")

        for breath in breaths.values:
            # Test whether the indices are in the proper order within a breath
            assert breath.start_time < breath.middle_time < breath.end_time

            # Test whether the peak values are larger than valley values
            assert cd.t[breath.middle_time].values[0] > cd.t[breath.start_time].values[0]
            assert cd.t[breath.middle_time].values[0] > cd.t[breath.end_time].values[0]

        start_indices, middle_indices, end_indices = (list(x) for x in zip(*breaths.values, strict=True))

        # Test whether breaths are sorted properly
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
            start_index >= end_index for start_index, end_index in zip(start_indices[1:], end_indices[:-1], strict=True)
        )


def test_create_breaths_from_peak_valley_data():
    time = np.arange(100) / 3.1415
    peak_indices = np.array([10, 20, 30, 40, 50])
    valley_indices = np.array([5, 15, 25, 35, 45, 55])

    bd = BreathDetection(1)
    breaths = bd._create_breaths_from_peak_valley_data(time, peak_indices, valley_indices)
    assert len(breaths) == 5
    assert all(isinstance(breath, Breath) for breath in breaths)
    assert np.array_equal(np.array([breath.start_time for breath in breaths]), time[valley_indices[:-1]])
    assert np.array_equal(np.array([breath.middle_time for breath in breaths]), time[peak_indices])
    assert np.array_equal(np.array([breath.end_time for breath in breaths]), time[valley_indices[1:]])

    fewer_valley_indices = valley_indices[:-1]
    with pytest.raises(ValueError):
        bd._create_breaths_from_peak_valley_data(time, peak_indices, fewer_valley_indices)

    fewer_peak_indices = peak_indices[:-1]
    with pytest.raises(ValueError):
        bd._create_breaths_from_peak_valley_data(time, fewer_peak_indices, valley_indices)

    peaks_out_of_order = np.concatenate([peak_indices[3:], peak_indices[:3]])
    with pytest.raises(ValueError):
        bd._create_breaths_from_peak_valley_data(time, peaks_out_of_order, valley_indices)

    valleys_out_of_order = np.concatenate([valley_indices[3:], valley_indices[:3]])
    with pytest.raises(ValueError):
        bd._create_breaths_from_peak_valley_data(time, peak_indices, valleys_out_of_order)

    peaks_at_same_index_as_valleys = valley_indices[:-1]
    with pytest.raises(ValueError):
        bd._create_breaths_from_peak_valley_data(time, peaks_at_same_index_as_valleys, valley_indices)


def test_remove_breaths_around_invalid_data():
    sample_frequency = 10
    length = 100
    frequency = 1
    time, y = make_cosine_wave(sample_frequency, length, frequency)

    peak_indices = np.arange(10, 100, 10)
    valley_indices = np.arange(5, 100, 10)

    assert np.array_equal(y[peak_indices], np.array([1.0] * 9))
    assert np.array_equal(y[valley_indices], np.array([-1.0] * 10))

    bd = BreathDetection(sample_frequency, invalid_data_removal_window_length=0.4)
    breaths = bd._create_breaths_from_peak_valley_data(time, peak_indices, valley_indices)

    outliers = np.array([], dtype=int)
    no_breaths_removed = bd._remove_breaths_around_invalid_data(breaths, time, outliers)
    assert no_breaths_removed == breaths
    assert no_breaths_removed is not breaths

    # a single outlier at a peak at t=6 should remove only breaths within 5.6 < t < 6.4
    outliers = np.array([60])
    breaths_with_some_removed = bd._remove_breaths_around_invalid_data(breaths, time, outliers)
    removed_breath = Breath(5.5, 6, 6.5)
    assert len(breaths_with_some_removed) == len(breaths) - 1
    assert removed_breath in breaths
    assert removed_breath not in breaths_with_some_removed  # the expected breaths were removed
    assert all(breath in breaths for breath in breaths_with_some_removed)  # all other stayed the same

    # a single outlier at a valley at t=6.5 should remove breaths overlapping with 6.1 < t < 6.9
    outliers = np.array([65])
    breaths_with_some_removed = bd._remove_breaths_around_invalid_data(breaths, time, outliers)
    removed_breaths = [Breath(5.5, 6, 6.5), Breath(6.5, 7, 7.5)]
    assert len(breaths_with_some_removed) == len(breaths) - 2
    assert all(removed_breath in breaths for removed_breath in removed_breaths)
    assert all(removed_breath not in breaths_with_some_removed for removed_breath in removed_breaths)
    assert all(kept_breath in breaths for kept_breath in breaths_with_some_removed)  # all other stayed the same