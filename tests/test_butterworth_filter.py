"""Tests for the Butterworth time domain filter"""
import numpy as np
import pytest
from scipy import signal
from eitprocessing.filters.butterworth_filters import BandPassFilter
from eitprocessing.filters.butterworth_filters import BandStopFilter
from eitprocessing.filters.butterworth_filters import ButterworthFilter
from eitprocessing.filters.butterworth_filters import HighPassFilter
from eitprocessing.filters.butterworth_filters import LowPassFilter


INIT_KWARGS = {
    "filter_type": "lowpass",
    "cutoff_frequency": 10,
    "sample_frequency": 50.2,
    "order": 5,
}


def check_filter_attributes(filter_, kwargs):
    for key, value in kwargs.items():
        assert getattr(filter_, key) == value

    assert hasattr(filter_, "apply_filter")
    assert callable(filter_.apply_filter)
    assert filter_.available_in_gui


def test_create_butterworth_filter():
    kwargs = INIT_KWARGS.copy()
    filter_ = ButterworthFilter(**kwargs)
    check_filter_attributes(filter_, kwargs)


def test_butterworth_range():
    # Tests whether an out-of-range order raises AttributeError
    kwargs = INIT_KWARGS.copy()
    del kwargs["order"]

    with pytest.raises(ValueError):
        ButterworthFilter(**kwargs, order=11)

    # Tests whether an explicit out-of-range order does not raise AttributeError
    try:
        filter_ = ButterworthFilter(**kwargs, order=11, override_order=True)
    except AttributeError:
        pytest.fail("Unexpected AttributeError")
    assert filter_.order == 11


def test_butterworth_filter_type():
    kwargs = INIT_KWARGS.copy()
    kwargs["filter_type"] = "invalid"
    with pytest.raises(ValueError):
        ButterworthFilter(**kwargs)


def test_butterworth_cutoff_frequency():
    kwargs = INIT_KWARGS.copy()
    del kwargs["cutoff_frequency"]

    with pytest.raises(TypeError):
        ButterworthFilter(**kwargs, cutoff_frequency="not a number")

    kwargs["filter_type"] = "bandpass"
    with pytest.raises(TypeError):
        ButterworthFilter(**kwargs, cutoff_frequency="not a number")

    with pytest.raises(TypeError):
        ButterworthFilter(**kwargs, cutoff_frequency=10)

    with pytest.raises(TypeError):
        ButterworthFilter(**kwargs, cutoff_frequency=("not a number", True))

    with pytest.raises(TypeError):
        ButterworthFilter(**kwargs, cutoff_frequency=[20, 30])

    with pytest.raises(ValueError):
        ButterworthFilter(**kwargs, cutoff_frequency=(1,))

    with pytest.raises(ValueError):
        ButterworthFilter(**kwargs, cutoff_frequency=(1, 2, 3))

    try:
        ButterworthFilter(**kwargs, cutoff_frequency=(20, 30))
    except (ValueError, TypeError):
        pytest.fail("Unexpected error")


def test_butterworth_sample_frequency():
    kwargs = INIT_KWARGS.copy()
    del kwargs["sample_frequency"]

    with pytest.raises(TypeError):
        ButterworthFilter(**kwargs, sample_frequency="a string")

    with pytest.raises(ValueError):
        ButterworthFilter(**kwargs, sample_frequency=-1)

    try:
        ButterworthFilter(**kwargs, sample_frequency=1)
    except (TypeError, ValueError):
        pytest.fail("Unexpected error")


def test_create_specified_filter():
    kwargs = INIT_KWARGS.copy()
    lp_filter = LowPassFilter(**kwargs)

    with pytest.raises(AttributeError):
        hp_filter = HighPassFilter(**kwargs)

    del kwargs["filter_type"]

    lp_filter = LowPassFilter(**kwargs)
    hp_filter = HighPassFilter(**kwargs)

    for filter_ in (lp_filter, hp_filter):
        check_filter_attributes(filter_, kwargs)

    kwargs["cutoff_frequency"] = (20, 30)
    bp_filter = BandPassFilter(**kwargs)
    bs_filter = BandStopFilter(**kwargs)

    for filter_ in (bp_filter, bs_filter):
        check_filter_attributes(filter_, kwargs)

    assert lp_filter.filter_type == "lowpass"
    assert hp_filter.filter_type == "highpass"
    assert bp_filter.filter_type == "bandpass"
    assert bs_filter.filter_type == "bandstop"


def test_specified_butterworth_equivalence():
    kwargs = INIT_KWARGS.copy()
    del kwargs["filter_type"]

    filter1 = ButterworthFilter(**kwargs, filter_type="lowpass")
    filter2 = LowPassFilter(**kwargs)
    assert filter1 == filter2
    assert (
        filter2 == filter1
    )  # filter1.__eq__(filter2) differs from filter2.__eq__(filter1)

    filter3 = ButterworthFilter(**kwargs, filter_type="highpass")
    filter4 = HighPassFilter(**kwargs)
    assert filter1 != filter4
    assert filter2 != filter4
    assert filter3 == filter4

    kwargs["cutoff_frequency"] = (20, 30)
    filter5 = ButterworthFilter(**kwargs, filter_type="bandpass")
    filter6 = BandPassFilter(**kwargs)
    assert filter1 != filter6
    assert filter2 != filter6
    assert filter5 == filter6

    filter7 = ButterworthFilter(**kwargs, filter_type="bandstop")
    filter8 = BandStopFilter(**kwargs)
    assert filter1 != filter8
    assert filter2 != filter8
    assert filter7 == filter8


def test_butterworth_functionality():
    sample_frequency = 50
    freq_low = 1
    freq_medium = 4
    freq_high = 10
    amplitude_medium = 0.5
    amplitude_high = 0.1

    order = 4

    t = np.arange(0, 100, 1 / sample_frequency)
    low_part = np.sin(2 * np.pi * t * freq_low)
    medium_part = np.sin(2 * np.pi * t * freq_medium)
    high_part = np.sin(2 * np.pi * t * freq_high)
    signal_ = low_part + amplitude_medium * medium_part + amplitude_high * high_part

    def compare_filters(cutoff, filter_type, class_):
        filter1 = ButterworthFilter(filter_type, cutoff, order, sample_frequency)
        filter2 = class_(cutoff, order, sample_frequency)
        result1 = filter1.apply_filter(signal_)
        result2 = filter2.apply_filter(signal_)
        assert np.array_equal(result1, result2)

        b, a = signal.butter(order, cutoff, filter_type, fs=sample_frequency)
        sp_result = signal.filtfilt(b, a, signal_)
        assert np.array_equal(result1, sp_result)

    lowpass_cutoff = (freq_low + freq_medium) / 2
    highpass_cutoff = (freq_medium + freq_high) / 2
    compare_filters(lowpass_cutoff, "lowpass", LowPassFilter)
    compare_filters(highpass_cutoff, "highpass", HighPassFilter)
    compare_filters((lowpass_cutoff, highpass_cutoff), "bandpass", BandPassFilter)
    compare_filters((lowpass_cutoff, highpass_cutoff), "bandstop", BandStopFilter)
