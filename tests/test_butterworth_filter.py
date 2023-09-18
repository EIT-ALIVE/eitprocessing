"""Tests for the Butterworth time domain filter"""
from typing import Literal
from typing import TypeAlias
import numpy as np
import pytest
from scipy import signal
from eitprocessing.filters.butterworth_filters import BandPassFilter
from eitprocessing.filters.butterworth_filters import BandStopFilter
from eitprocessing.filters.butterworth_filters import ButterworthFilter
from eitprocessing.filters.butterworth_filters import HighPassFilter
from eitprocessing.filters.butterworth_filters import LowPassFilter


SpecifiedFilter: TypeAlias = type[
    LowPassFilter | HighPassFilter | BandPassFilter | BandStopFilter
]


@pytest.fixture
def filter_arguments():
    return {
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


def test_create_butterworth_filter(filter_arguments):
    filter_ = ButterworthFilter(**filter_arguments)
    check_filter_attributes(filter_, filter_arguments)


def test_butterworth_range(filter_arguments):
    # Tests whether an out-of-range order raises AttributeError
    filter_arguments["order"] = 11

    with pytest.raises(ValueError):
        ButterworthFilter(**filter_arguments)

    # Tests whether an explicit out-of-range order does not raise AttributeError
    try:
        filter_ = ButterworthFilter(**filter_arguments, ignore_max_order=True)
    except AttributeError:
        pytest.fail("Unexpected AttributeError")
    assert filter_.order == 11


def test_butterworth_filter_type(filter_arguments):
    filter_arguments["filter_type"] = "invalid"
    with pytest.raises(ValueError):
        ButterworthFilter(**filter_arguments)


def test_butterworth_cutoff_frequency_scalar(filter_arguments):
    del filter_arguments["cutoff_frequency"]

    invalid_bandpass_cutoffs = [("not a number", TypeError), ((10, 20), TypeError)]
    for invalid, error_type in invalid_bandpass_cutoffs:
        with pytest.raises(error_type):
            ButterworthFilter(**filter_arguments, cutoff_frequency=invalid)

    try:
        ButterworthFilter(**filter_arguments, cutoff_frequency=20)
    except (ValueError, TypeError):
        pytest.fail("Unexpected error")


def test_butterworth_cutoff_frequency_sequence(filter_arguments):
    del filter_arguments["cutoff_frequency"]
    filter_arguments["filter_type"] = "bandpass"

    invalid_bandpass_cutoffs = [
        ("not a number", TypeError),
        (10, TypeError),
        (("not a number", True), TypeError),
        ((1, "not a number"), TypeError),
        ((1,), ValueError),
        ((1, 2, 3), ValueError),
    ]
    for invalid, error_type in invalid_bandpass_cutoffs:
        with pytest.raises(error_type):
            ButterworthFilter(**filter_arguments, cutoff_frequency=invalid)

    try:
        ButterworthFilter(**filter_arguments, cutoff_frequency=(20, 30))
    except (ValueError, TypeError):
        pytest.fail("Unexpected error")


def test_butterworth_sample_frequency(filter_arguments):
    del filter_arguments["sample_frequency"]

    with pytest.raises(TypeError):
        ButterworthFilter(**filter_arguments, sample_frequency="a string")

    with pytest.raises(ValueError):
        ButterworthFilter(**filter_arguments, sample_frequency=-1)

    try:
        ButterworthFilter(**filter_arguments, sample_frequency=1)
    except (TypeError, ValueError):
        pytest.fail("Unexpected error")


def test_create_specified_filter(filter_arguments):
    lp_filter = LowPassFilter(**filter_arguments)

    with pytest.raises(TypeError):
        hp_filter = HighPassFilter(**filter_arguments)

    del filter_arguments["filter_type"]

    lp_filter = LowPassFilter(**filter_arguments)
    hp_filter = HighPassFilter(**filter_arguments)

    for filter_ in (lp_filter, hp_filter):
        check_filter_attributes(filter_, filter_arguments)

    filter_arguments["cutoff_frequency"] = (20, 30)
    bp_filter = BandPassFilter(**filter_arguments)
    bs_filter = BandStopFilter(**filter_arguments)

    for filter_ in (bp_filter, bs_filter):
        check_filter_attributes(filter_, filter_arguments)

    assert lp_filter.filter_type == "lowpass"
    assert hp_filter.filter_type == "highpass"
    assert bp_filter.filter_type == "bandpass"
    assert bs_filter.filter_type == "bandstop"


def test_specified_butterworth_equivalence(filter_arguments):
    del filter_arguments["filter_type"]

    filter1 = ButterworthFilter(**filter_arguments, filter_type="lowpass")
    filter2 = LowPassFilter(**filter_arguments)

    # filter1.__eq__(filter2) differs from filter2.__eq__(filter1)
    assert filter1 == filter2
    assert filter2 == filter1

    filter3 = ButterworthFilter(**filter_arguments, filter_type="highpass")
    filter4 = HighPassFilter(**filter_arguments)
    assert filter1 != filter4
    assert filter2 != filter4
    assert filter3 == filter4

    filter_arguments["cutoff_frequency"] = (20, 30)
    filter5 = ButterworthFilter(**filter_arguments, filter_type="bandpass")
    filter6 = BandPassFilter(**filter_arguments)
    assert filter1 != filter6
    assert filter2 != filter6
    assert filter5 == filter6

    filter7 = ButterworthFilter(**filter_arguments, filter_type="bandstop")
    filter8 = BandStopFilter(**filter_arguments)
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
    lowpass_cutoff = (freq_low + freq_medium) / 2
    highpass_cutoff = (freq_medium + freq_high) / 2

    order = 4

    t = np.arange(0, 100, 1 / sample_frequency)
    low_part = np.sin(2 * np.pi * t * freq_low)
    medium_part = np.sin(2 * np.pi * t * freq_medium)
    high_part = np.sin(2 * np.pi * t * freq_high)
    signal_ = low_part + amplitude_medium * medium_part + amplitude_high * high_part

    def compare_filters(
        cutoff: float | tuple[float, float],
        filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"],
        class_: SpecifiedFilter,
    ):
        """Compare filters created using ButterworthFilter to filters created using the
        corresponding subclass

        This function creates two filter instances, one using the ButterworthFilter, and one using
        either of the four subclasses. It then compares whether those filters are equal, and have
        equal results.

        The function then filters the data `signal_` using the created ButterworthFilter, and
        compares the result to the same data filtered using `scipy`.

        Parameters:
            cutoff: the cutoff frequency to use
            filter_type: the filter type to use
            class_: the class corresponding to the filter_type

        """
        filter1 = ButterworthFilter(
            filter_type=filter_type,
            cutoff_frequency=cutoff,
            order=order,
            sample_frequency=sample_frequency,
        )
        filter2 = class_(
            cutoff_frequency=cutoff, order=order, sample_frequency=sample_frequency
        )
        result1 = filter1.apply_filter(signal_)
        result2 = filter2.apply_filter(signal_)
        assert np.array_equal(result1, result2)

        sos = signal.butter(
            order, cutoff, filter_type, fs=sample_frequency, output="sos"
        )
        sp_result = signal.sosfiltfilt(sos, signal_)
        assert np.array_equal(result1, sp_result)

    compare_filters(lowpass_cutoff, "lowpass", LowPassFilter)
    compare_filters(highpass_cutoff, "highpass", HighPassFilter)
    compare_filters((lowpass_cutoff, highpass_cutoff), "bandpass", BandPassFilter)
    compare_filters((lowpass_cutoff, highpass_cutoff), "bandstop", BandStopFilter)