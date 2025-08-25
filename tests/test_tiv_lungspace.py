from collections.abc import Callable

import numpy as np
import pytest
from numpy import typing as npt

from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.roi.tiv import TIVLungspace


@pytest.fixture
def create_signal():
    """Fixture to create a simple EIT signal with given amplitudes for testing.

    The resulting EITData object has a pixel impedance signal of which the last two dimensions are the same as the shape
    of the provided amplitudes array. The signal is a sine wave with a period of 5 seconds. The amplitude of each pixel
    is determined by the corresponding value in the amplitudes array. Before multiplication with the amplitude, 0.5 was
    added to the sine wave, such that 75% of the signal amplitude is >=0, and 25% is <=0.
    """

    def factory(amplitudes: npt.ArrayLike, duration: float = 20.0) -> EITData:
        amplitudes = np.asarray(amplitudes)
        sample_frequency = 20
        t = np.arange(int(duration * sample_frequency)) / sample_frequency
        sine = np.sin(t * 2 * np.pi / 5)  # 5 second period
        signal = amplitudes[None, :, :] * (sine[:, None, None] + 0.5)
        return EITData(
            pixel_impedance=signal,
            path="",
            nframes=len(t),
            time=t,
            sample_frequency=sample_frequency,
            vendor=Vendor.SIMULATED,
            label="simulated",
            description="Simulated EIT data for testing purposes",
            name="",
            suppress_simulated_warning=True,
        )

    return factory


def test_tiv_lungspace_init():
    _ = TIVLungspace()
    _ = TIVLungspace(threshold=0.2)
    _ = TIVLungspace(mode="amplitude")
    _ = TIVLungspace(threshold=0.5, mode="amplitude")
    _ = TIVLungspace(threshold=0.01, mode="TIV")


@pytest.mark.parametrize("threshold", [-0.1, 0.0, 1.0, 1.5])
def test_tiv_lungspace_init_threshold_outside_range(threshold: float):
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1."):
        _ = TIVLungspace(threshold=threshold)


@pytest.mark.parametrize("threshold", ["0.2", True, None])
def test_tiv_lungspace_init_threshold_wrong_type(threshold: object):
    with pytest.raises(TypeError, match="Threshold must be a float."):
        _ = TIVLungspace(threshold=threshold)


@pytest.mark.parametrize("mode", [123, True, None, "normal"])
def test_tiv_lungspace_init_mode_wrong_type(mode: object):
    with pytest.raises(ValueError, match="Unknown mode '.+'. Supported modes are 'TIV' and 'amplitude'."):
        _ = TIVLungspace(mode=mode)


def test_tiv_lungspace_apply(create_signal: Callable):
    amplitudes = np.array([[-1.0, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    signal = create_signal(amplitudes, duration=60)
    tiv_lungspace = TIVLungspace(threshold=0.15, mode="TIV").apply(signal, captures=(captures_tiv := {}))

    assert tiv_lungspace.values.shape == amplitudes.shape
    assert np.allclose(captures_tiv["mean TIV"].values, [[-2.0, -1.0, 0, 0.4, 0.8, 1.2, 1.6, 2.0]])
    assert np.array_equal(
        tiv_lungspace.values,
        np.array([[np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        equal_nan=True,
    )

    ampl_lungspace = TIVLungspace(threshold=0.15, mode="amplitude").apply(signal, captures=(captures_ampl := {}))
    assert ampl_lungspace.values.shape == amplitudes.shape
    assert np.allclose(
        captures_ampl["mean amplitude"].values, [[2.0, 1.0, np.nan, 0.4, 0.8, 1.2, 1.6, 2.0]], equal_nan=True
    )
    assert np.array_equal(
        ampl_lungspace.values,
        np.array([[1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        equal_nan=True,
    )


def test_tiv_lungspace_apply_no_breaths(create_signal: Callable):
    amplitudes = np.zeros((1, 8))
    signal = create_signal(amplitudes, duration=60)

    with pytest.raises(ValueError, match="No breaths were detected. Cannot compute TIV"):
        _ = TIVLungspace(threshold=0.2, mode="TIV").apply(signal)

    with pytest.raises(ValueError, match="No breaths were detected. Cannot compute amplitude"):
        _ = TIVLungspace(threshold=0.2, mode="amplitude").apply(signal)


def test_tiv_lungspace_apply_no_non_nan_values(create_signal: Callable):
    amplitudes = np.full((1, 8), 1)
    signal = create_signal(amplitudes, duration=15)

    # Works with TIV, because it needs only one detected breath
    _ = TIVLungspace(threshold=0.2, mode="TIV").apply(signal)

    # Does not work with amplitude, because it needs three detected breaths
    with pytest.raises(ValueError, match="No non-nan amplitude were found."):
        _ = TIVLungspace(threshold=0.2, mode="amplitude").apply(signal)


def test_tiv_lungspace_with_timing_data(create_signal: Callable):
    amplitudes = np.ones((1, 8))
    signal = create_signal(amplitudes, duration=60)
    timing_data = signal.get_summed_impedance()

    _ = TIVLungspace(threshold=0.2, mode="TIV").apply(signal, timing_data=timing_data)


def test_tiv_lungpsace_with_real_data(draeger1: Sequence):
    eit_data = draeger1.eit_data["raw"]
    _ = TIVLungspace(threshold=0.2, mode="TIV").apply(eit_data)
    _ = TIVLungspace(threshold=0.2, mode="amplitude").apply(eit_data)
