# %%

import itertools
import warnings
from collections.abc import Callable

import numpy as np
import pytest
from scipy import signal

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.rate_detection import DEFAULT_RATE_LIMITS, RateDetection
from eitprocessing.plotting.rate_detection import RateDetectionPlotting

MINUTE = 60


@pytest.fixture
def signal_factory() -> Callable[..., EITData]:
    """Create a factory function to generate synthetic EIT data with specified characteristics.

    The generated EIT data is suitable for testing the RateDetection algorithm. It includes high and low power frequency
    components, noise, and allows for customization of parameters such as duration, sample frequency, shape, and
    amplitudes. The high and low power signals are generated using triangular waveforms.

    The high power signal has high power in half the pixels, and lower power in the other half. This simulates a limited
    number of pixels with mainly respiratory signal. The low power signal has lower power in all pixels, simulating a
    more uniform distribution of heart rate signal. In most data, a single high power frequency and single low power
    frequency should be used, but the factory allows for multiple frequencies to be specified to simulate shifting or
    irregular signals. The exact amplitude of each frequency component is randomized to add variability to the generated
    data.

    The frequencies can be increased or decreased from the start to the end of the signal. The frequency at the end of
    the signal is the base frequency multiplied by the frequency scale factor. A frequency scale factor of 1.0 will
    result in a constant frequency, values greater than 1.0 will lead to an increasing frequency, and values less than
    1.0 will lead to a decreasing frequency.

    The first row of the pixel impedance is set to NaN to simulatte missing data.

    Arguments:
        high_power_frequencies: Frequencies for the high power signal in Hz.
        low_power_frequencies: Frequencies for the low power signal in Hz.
        duration: Duration of the signal in seconds (default is 60.0).
        sample_frequency: Sample frequency in Hz (default is 20.0).
        shape: Shape of the pixel impedance array (default is (32, 32)).
        low_power_amplitude: Amplitude of the low power signal (default is 0.2).
        noise_amplitude: Amplitude of the noise added to the signal (default is 0.01).
        captures: Optional dictionary to capture generated signals and time for testing purposes.
    """

    def factory(  # noqa: PLR0913
        *,
        high_power_frequencies: tuple[float, ...],
        low_power_frequencies: tuple[float, ...],
        duration: float = 60.0,
        sample_frequency: float = 20.0,
        shape: tuple[int, int] = (32, 32),
        low_power_amplitude: float = 0.2,
        noise_amplitude: float = 0.01,
        high_frequency_scale_factor: float = 1.0,
        low_frequency_scale_factor: float = 1.0,
        captures: dict | None = None,
    ) -> EITData:
        nframes = int(duration * sample_frequency)
        time = np.arange(nframes) / sample_frequency
        rng = np.random.default_rng()

        # Generate base signals
        high_power_signal = np.zeros((nframes,))
        low_power_signal = np.zeros((nframes,))

        for freq in high_power_frequencies:
            inst_freq = np.linspace(freq, freq * high_frequency_scale_factor, len(time))
            phase = 2 * np.pi * np.cumsum(inst_freq) / sample_frequency
            high_power_signal += rng.normal(loc=1, scale=0.1) * signal.sawtooth(phase, width=0.5)
        for freq in low_power_frequencies:
            inst_freq = np.linspace(freq, freq * low_frequency_scale_factor, len(time))
            phase = 2 * np.pi * np.cumsum(inst_freq) / sample_frequency
            low_power_signal += rng.normal(loc=1, scale=0.1) * signal.sawtooth(phase, width=0.5)

        # S-shaped mask for high power (along columns)
        x = np.linspace(-6, 6, shape[1])
        s_mask = 1 / (1 + np.exp(-x))  # sigmoid, shape (shape[1],)
        s_mask = s_mask[None, :]  # shape (1, shape[1])
        s_mask = np.tile(s_mask, (shape[0], 1))  # shape (shape[0], shape[1])

        # High power: S-shaped mask, low power: 1 - mask
        high_power_pixels = high_power_signal[:, None, None] * s_mask[None, :, :]
        # Low power: random noise around mean (e.g., mean=0.2, std=0.05)

        low_power_pixels = (
            low_power_signal[:, None, None] * rng.normal(loc=low_power_amplitude, scale=0.05, size=shape)[None, :, :]
        )

        noise = rng.normal(loc=0, scale=noise_amplitude, size=(nframes, *shape))

        pixel_impedance = high_power_pixels + low_power_pixels + noise
        pixel_impedance[:, 0, :] = np.nan

        if captures is not None:
            captures["high_power_signal"] = high_power_signal
            captures["low_power_signal"] = low_power_signal
            captures["time"] = time
            captures["pixel_impedance"] = pixel_impedance
            captures["summed_impedance"] = np.nansum(pixel_impedance, axis=(1, 2))

        return EITData(
            path=".",
            nframes=nframes,
            time=time,
            sample_frequency=sample_frequency,
            vendor="draeger",
            label="test_signal",
            pixel_impedance=pixel_impedance,
        )

    return factory


def test_init_rate_detection():
    """Test the initialization of the RateDetection class."""
    rd = RateDetection("adult")
    assert rd.subject_type == "adult"
    assert rd.refine_estimated_frequency is True
    assert rd.welch_window == 30.0
    assert rd.welch_overlap == 0.5
    assert isinstance(rd.plotting, RateDetectionPlotting)

    with pytest.raises(ValueError, match="Invalid subject type"):
        _ = RateDetection("dolphin")

    with pytest.raises(ValueError, match="Welch overlap must be less than 1.0"):
        _ = RateDetection("adult", welch_overlap=1.0)

    with pytest.raises(ValueError, match="Welch overlap must be less than 1.0"):
        _ = RateDetection("adult", welch_overlap=20)

    with pytest.raises(ValueError, match="Welch overlap must be at least 0"):
        _ = RateDetection("adult", welch_overlap=-0.1)

    with pytest.raises(ValueError, match="Welch window must be at least 10 seconds."):
        _ = RateDetection("adult", welch_window=9)


def test_short_signal(signal_factory: Callable[..., EITData]):
    rd = RateDetection("adult", welch_window=30.0)

    short_signal = signal_factory(
        high_power_frequencies=(0.25,),
        low_power_frequencies=(1.75,),
        duration=15,
    )
    with pytest.warns(UserWarning, match="The Welch window is longer than the data. Reducing the window length"):
        _ = rd.apply(short_signal)

    long_signal = signal_factory(
        high_power_frequencies=(0.25,),
        low_power_frequencies=(1.75,),
        duration=30,
    )
    with warnings.catch_warnings(record=True) as w:
        rd.apply(long_signal)
        assert len(w) == 0

    rd_short = RateDetection("adult", welch_window=10)
    with warnings.catch_warnings(record=True) as w:
        rd_short.apply(long_signal)
        rd_short.apply(short_signal)
        assert len(w) == 0


high_power_frequencies = np.linspace(
    DEFAULT_RATE_LIMITS["min_respiratory_rate"]["adult"], DEFAULT_RATE_LIMITS["max_respiratory_rate"]["adult"], 9
)[2:-2]
low_power_frequencies = np.linspace(
    DEFAULT_RATE_LIMITS["min_heart_rate"]["adult"], DEFAULT_RATE_LIMITS["max_heart_rate"]["adult"], 9
)[2:-2]
duration = np.linspace(10, 60, 5)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("high_power_frequency", "low_power_frequency", "duration"),
    list(itertools.product(high_power_frequencies, low_power_frequencies, duration)),
)
def test_frequencies(
    high_power_frequency: float, low_power_frequency: float, duration: float, signal_factory: Callable[..., EITData]
) -> None:
    """Test the RateDetection algorithm with varying high and low power frequencies."""
    if high_power_frequency >= low_power_frequency * 0.8:
        pytest.skip(r"High power frequency must be lower than 80\% of the low power frequency for this test.")
    signal = signal_factory(
        high_power_frequencies=(high_power_frequency,),
        low_power_frequencies=(low_power_frequency,),
        duration=duration,
        sample_frequency=20.0,
        shape=(32, 32),
        low_power_amplitude=0.01,
        noise_amplitude=0.1,
    )

    rd = RateDetection("adult", refine_estimated_frequency=True)
    refined_rr, refined_hr = rd.apply(signal, suppress_edge_case_warning=True, suppress_length_warnings=True)

    rd = RateDetection("adult", refine_estimated_frequency=False)
    non_refined_rr, non_refined_hr = rd.apply(signal, suppress_edge_case_warning=True, suppress_length_warnings=True)

    assert np.isclose(refined_rr, high_power_frequency, rtol=0.05, atol=0.5 / MINUTE)
    assert np.isclose(refined_hr, low_power_frequency, rtol=0.05, atol=0.5 / MINUTE)

    assert np.isclose(non_refined_rr, high_power_frequency, rtol=0.2, atol=1 / MINUTE)
    assert np.isclose(non_refined_hr, low_power_frequency, rtol=0.2, atol=1 / MINUTE)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("high_power_frequency", "harmonic_multiplier", "duration"),
    list(itertools.product(high_power_frequencies, range(2, 5), duration)),
)
def test_harmonic_heart_rate(
    high_power_frequency: float,
    harmonic_multiplier: int,
    duration: float,
    signal_factory: Callable[..., EITData],
):
    """Test the RateDetection algorithm with a harmonic heart rate."""
    low_power_frequency = high_power_frequency * harmonic_multiplier
    if (
        low_power_frequency >= DEFAULT_RATE_LIMITS["max_heart_rate"]["adult"]
        or low_power_frequency <= DEFAULT_RATE_LIMITS["min_heart_rate"]["adult"]
    ):
        pytest.skip("Low power frequency is outside the heart rate range for this test.")

    signal = signal_factory(
        high_power_frequencies=(high_power_frequency,),
        low_power_frequencies=(low_power_frequency,),
        duration=duration,
        sample_frequency=20.0,
        shape=(32, 32),
        low_power_amplitude=0.01,
        noise_amplitude=0.1,
    )

    rd = RateDetection("adult", refine_estimated_frequency=True)
    rr, hr = rd.apply(signal, suppress_edge_case_warning=True, suppress_length_warnings=True)

    assert np.isclose(rr, high_power_frequency, rtol=0.05, atol=0.5 / MINUTE)
    assert np.isclose(hr, low_power_frequency, rtol=0.05, atol=0.5 / MINUTE)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("high_power_frequency", "low_power_frequency", "frequency_multipliers"),
    list(itertools.product(high_power_frequencies, low_power_frequencies, [(1, 1.3), (1, 1.4)])),
)
def test_multiple_frequencies(
    signal_factory: Callable[..., EITData],
    high_power_frequency: float,
    low_power_frequency: float,
    frequency_multipliers: tuple[float, ...],
) -> None:
    """Test the RateDetection algorithm with multiple high and low power frequencies."""
    high_power_frequencies = tuple(high_power_frequency * m for m in frequency_multipliers)
    low_power_frequencies = tuple(low_power_frequency * m for m in frequency_multipliers)

    if any(
        f < DEFAULT_RATE_LIMITS["min_respiratory_rate"]["adult"]
        or f > DEFAULT_RATE_LIMITS["max_respiratory_rate"]["adult"]
        for f in high_power_frequencies
    ) or any(
        f < DEFAULT_RATE_LIMITS["min_heart_rate"]["adult"] or f > DEFAULT_RATE_LIMITS["max_heart_rate"]["adult"]
        for f in low_power_frequencies
    ):
        pytest.skip("One or more frequencies are outside the respiratory rate range for this test.")

    signal = signal_factory(
        high_power_frequencies=high_power_frequencies,
        low_power_frequencies=low_power_frequencies,
        duration=60.0,
        sample_frequency=20.0,
        shape=(32, 32),
        low_power_amplitude=0.01,
        noise_amplitude=0.1,
    )

    rd = RateDetection("adult", refine_estimated_frequency=True)
    rr, hr = rd.apply(signal, suppress_edge_case_warning=True, suppress_length_warnings=True)

    assert np.any(np.isclose(rr, high_power_frequencies, rtol=0.05, atol=0.5 / MINUTE))
    assert np.any(np.isclose(hr, low_power_frequencies, rtol=0.05, atol=0.5 / MINUTE))


scale_factors = (0.8, 0.9, 1.0, 1.1, 1.2)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("high_power_frequency", "low_power_frequency", "high_frequency_scale_factor", "low_frequency_scale_factor"),
    list(itertools.product(high_power_frequencies, low_power_frequencies, scale_factors, scale_factors)),
)
def test_changing_frequency(
    signal_factory: Callable[..., EITData],
    high_power_frequency: float,
    low_power_frequency: float,
    high_frequency_scale_factor: float,
    low_frequency_scale_factor: float,
):
    """Test the RateDetection algorithm with changing frequencies."""
    if high_power_frequency >= low_power_frequency * 0.8:
        pytest.skip(r"High power frequency must be lower than 80\% of the low power frequency for this test.")

    # Create a signal with a changing high power frequency
    signal = signal_factory(
        high_power_frequencies=(high_power_frequency,),
        low_power_frequencies=(low_power_frequency,),
        duration=60.0,
        sample_frequency=20.0,
        shape=(32, 32),
        low_power_amplitude=0.01,
        noise_amplitude=0.1,
        high_frequency_scale_factor=high_frequency_scale_factor,
        low_frequency_scale_factor=low_frequency_scale_factor,
    )

    rd = RateDetection("adult", refine_estimated_frequency=True)
    rr, hr = rd.apply(signal, suppress_edge_case_warning=True, suppress_length_warnings=True)

    assert np.isclose(rr, high_power_frequency * (1 + high_frequency_scale_factor) / 2, rtol=0.1, atol=0.5 / MINUTE)
    assert np.isclose(hr, low_power_frequency * (1 + low_frequency_scale_factor) / 2, rtol=0.1, atol=0.5 / MINUTE)


@pytest.mark.parametrize(
    ("sequence", "slice_", "expected_rr", "expected_hr"),
    [
        ("draeger1", slice(None, None), 0.124, 1.121),
        ("draeger2", slice(56650, 56760), 0.416, 1.183),
        ("timpel1", slice(None, None), 0.329, 2.196),
    ],
    indirect=["sequence"],
)
def test_with_data(sequence: Sequence, slice_: slice, expected_rr: float, expected_hr: float):
    """Test the RateDetection algorithm with real EIT data."""
    rd = RateDetection("adult")
    eit_data: EITData = sequence.eit_data["raw"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="No starting or end timepoint was selected.")
        sub_data = eit_data.t[slice_]

    rr, hr = rd.apply(sub_data)

    assert isinstance(rr, float)
    assert isinstance(hr, float)
    assert np.isclose(rr, expected_rr, rtol=0.05, atol=0.5 / MINUTE)
    assert np.isclose(hr, expected_hr, rtol=0.05, atol=0.5 / MINUTE)
