import itertools
from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
from scipy import signal

from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.filters.mdn import UPPER_HEART_RATE_LIMIT, UPPER_RESPIRATORY_RATE_LIMIT, MDNFilter

MINUTE = 60


@pytest.fixture
def periodogram() -> Callable:
    return partial(
        signal.periodogram,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )


@pytest.fixture
def signal_factory() -> Callable[..., np.ndarray]:
    def factory(
        *,
        high_power_frequencies: tuple[float, ...],
        low_power_frequencies: tuple[float, ...],
        duration: float = 60.0,
        sample_frequency: float = 20.0,
        low_power_amplitude: float = 0.2,
        noise_amplitude: float = 0.01,
        high_frequency_scale_factor: float = 1.0,
        low_frequency_scale_factor: float = 1.0,
        captures: dict | None = None,
    ) -> np.ndarray:
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

        noise = rng.normal(loc=0, scale=noise_amplitude, size=(nframes,))

        values = high_power_signal + low_power_amplitude * low_power_signal + noise

        if captures is not None:
            captures["high_power_signal"] = high_power_signal
            captures["low_power_signal"] = low_power_signal
            captures["time"] = time
            captures["values"] = values

        return values

    return factory


def test_respiratory_rate_above_limits():
    with pytest.warns(UserWarning, match=r"The provided respiratory rate \(.*\) is higher than .* Hz \(.* BPM\)"):
        _ = MDNFilter(
            respiratory_rate=UPPER_RESPIRATORY_RATE_LIMIT + 0.01,
            heart_rate=UPPER_HEART_RATE_LIMIT - 0.01,
        )


def test_heart_rate_above_limits():
    with pytest.warns(UserWarning, match=r"The provided heart rate \(.*\) is higher than .* Hz \(.* BPM\)"):
        _ = MDNFilter(
            respiratory_rate=UPPER_RESPIRATORY_RATE_LIMIT - 0.01,
            heart_rate=UPPER_HEART_RATE_LIMIT + 0.01,
        )


def test_negative_respiratory_rate():
    with pytest.raises(ValueError, match=r"The provided respiratory rate \(.*\) must be positive."):
        _ = MDNFilter(
            respiratory_rate=-1 / MINUTE,
            heart_rate=80 / MINUTE,
        )


def test_negative_heart_rate():
    with pytest.raises(ValueError, match=r"The provided heart rate \(.*\) must be positive."):
        _ = MDNFilter(
            respiratory_rate=10 / MINUTE,
            heart_rate=-80 / MINUTE,
        )


def test_respiratory_rate_higher_than_heart_rate():
    with pytest.raises(
        ValueError, match=r"The respiratory rate \(.* Hz\) is equal to or higher than the heart rate \(.* Hz\)"
    ):
        _ = MDNFilter(
            respiratory_rate=UPPER_RESPIRATORY_RATE_LIMIT - 0.01,
            heart_rate=UPPER_RESPIRATORY_RATE_LIMIT - 0.02,
        )


def test_with_continuous_data(draeger1: Sequence):
    continuous_data = draeger1.continuous_data["global_impedance_(raw)"]
    mdn_filter = MDNFilter(
        respiratory_rate=10 / MINUTE,
        heart_rate=80 / MINUTE,
    )

    filtered_data = mdn_filter.apply(continuous_data)
    filtered_signal = mdn_filter.apply(
        continuous_data.values, sample_frequency=continuous_data.sample_frequency, axis=0
    )

    assert np.allclose(filtered_data.values, filtered_signal)


def test_with_eit_data(draeger1: Sequence):
    eit_data = draeger1.eit_data["raw"]
    mdn_filter = MDNFilter(
        respiratory_rate=10 / MINUTE,
        heart_rate=80 / MINUTE,
    )

    filtered_data = mdn_filter.apply(eit_data)
    filtered_signal = mdn_filter.apply(eit_data.pixel_impedance, sample_frequency=eit_data.sample_frequency, axis=0)

    assert np.allclose(filtered_data.pixel_impedance, filtered_signal)


@pytest.mark.parametrize(
    ("respiratory_rate", "heart_rate", "sample_frequency", "order"),
    itertools.product(
        (10 / MINUTE, 20 / MINUTE, 30 / MINUTE), (80 / MINUTE, 120 / MINUTE, 160 / MINUTE), (20, 50.2), (1, 5, 10, 20)
    ),
)
def test_with_numpy_different_frequencies(
    signal_factory: Callable,
    periodogram: Callable,
    respiratory_rate: float,
    heart_rate: float,
    sample_frequency: float,
    order: int,
):
    signal = signal_factory(
        high_power_frequencies=(respiratory_rate,),
        low_power_frequencies=(heart_rate,),
        sample_frequency=sample_frequency,
        duration=120.0,
    )

    mdn_filter = MDNFilter(
        respiratory_rate=respiratory_rate,
        heart_rate=heart_rate,
        order=order,
    )

    filtered_signal = mdn_filter.apply(signal, sample_frequency=sample_frequency, captures=(captures := {}))

    frequencies, power_unfiltered = periodogram(signal)
    _, power_filtered = periodogram(filtered_signal)

    assert np.sum(power_filtered) < np.sum(power_unfiltered)

    # Within the filtered frequency bands, power must be lower
    for lower_freq, higher_freq in [*captures["frequency_bands"]]:
        slice_ = slice(*np.searchsorted(frequencies, [lower_freq, higher_freq]))
        assert np.all(power_filtered[slice_] < power_unfiltered[slice_])

    # Above the noise frequency limit, power must be lower
    slice_ = slice(np.searchsorted(frequencies, mdn_filter.noise_frequency_limit), None)
    assert np.all(power_filtered[slice_] < power_unfiltered[slice_])

    # The lower frequency of the first filtered band is the normal notch distance away
    assert captures["frequency_bands"][0][0] == heart_rate - mdn_filter.notch_distance

    assert len(captures["frequency_bands"]) == captures["n_harmonics"]


def test_sample_frequency_not_provided():
    signal = np.random.default_rng().normal(size=1000)
    mdn_filter = MDNFilter(
        respiratory_rate=10 / MINUTE,
        heart_rate=80 / MINUTE,
    )
    with pytest.raises(ValueError, match="Sample frequency must be provided."):
        mdn_filter.apply(signal)

    with pytest.raises(ValueError, match="Sample frequency must be provided."):
        mdn_filter.apply(signal, sample_frequency=None)


def test_respiratory_rate_and_heart_rate_equal():
    common_rate = 60 / MINUTE
    with pytest.raises(
        ValueError, match=r"The respiratory rate \(.* Hz\) is equal to or higher than the heart rate \(.* Hz\)"
    ):
        _ = MDNFilter(respiratory_rate=common_rate, heart_rate=common_rate)


def test_close_respiratory_and_heart_rate(signal_factory: Callable):
    heart_rate = 55 / MINUTE
    respiratory_rate = 40 / MINUTE
    signal = signal_factory(
        high_power_frequencies=(respiratory_rate,),
        low_power_frequencies=(heart_rate,),
        sample_frequency=50,
    )

    mdn_filter = MDNFilter(respiratory_rate=respiratory_rate, heart_rate=heart_rate)
    _ = mdn_filter.apply(signal, sample_frequency=50, captures=(captures := {}))

    assert captures["n_harmonics"] == 4

    # With the rates this close, the first band start exactly between them
    assert captures["frequency_bands"][0][0] == np.mean([respiratory_rate, heart_rate])

    mdn_filter = MDNFilter(respiratory_rate=respiratory_rate, heart_rate=heart_rate, notch_distance=5 / MINUTE)
    _ = mdn_filter.apply(signal, sample_frequency=50, captures=(captures := {}))

    # With a smaller notch distance, the lower frequency is the notch distance away from the heart rate
    assert captures["frequency_bands"][0][0] == heart_rate - 5 / MINUTE


def test_wrong_input_type_raises():
    mdn_filter = MDNFilter(
        respiratory_rate=10 / MINUTE,
        heart_rate=80 / MINUTE,
    )
    with pytest.raises(TypeError, match="Invalid input data type"):
        mdn_filter.apply("not a valid input type")

    with pytest.raises(TypeError, match="Invalid input data type"):
        mdn_filter.apply(12345)


def test_provide_sample_frequency_axis_with_datacontainers_raises(draeger1: Sequence):
    eit_data = draeger1.eit_data["raw"]
    continuous_data = draeger1.continuous_data["global_impedance_(raw)"]
    mdn_filter = MDNFilter(
        respiratory_rate=10 / MINUTE,
        heart_rate=80 / MINUTE,
    )

    with pytest.raises(ValueError, match="Sample frequency should not be provided"):
        mdn_filter.apply(continuous_data, sample_frequency=50)

    with pytest.raises(ValueError, match="Sample frequency should not be provided"):
        mdn_filter.apply(eit_data, sample_frequency=50)

    with pytest.raises(ValueError, match="Axis should not be provided"):
        mdn_filter.apply(continuous_data, axis=0)

    with pytest.raises(ValueError, match="Axis should not be provided"):
        mdn_filter.apply(eit_data, axis=0)


def test_kwargs(draeger1: Sequence):
    eit_data = draeger1.eit_data["raw"]
    mdn_filter = MDNFilter(
        respiratory_rate=10 / MINUTE,
        heart_rate=80 / MINUTE,
    )

    # Ensure that kwargs are passed through correctly
    filtered_data = mdn_filter.apply(eit_data, label="Filtered EIT Data")

    assert filtered_data.label == "Filtered EIT Data"


def test_plot_filter_effects(draeger1: Sequence):
    """This test only checks that the plotting function runs without error."""
    impedance = draeger1.continuous_data["global_impedance_(raw)"]
    mdn_filter = MDNFilter(
        respiratory_rate=10 / MINUTE,
        heart_rate=80 / MINUTE,
    )
    mdn_filter.apply(impedance, captures=(captures := {}))
    mdn_filter.plotting.plot_results(**captures)
