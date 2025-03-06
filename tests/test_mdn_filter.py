import numpy as np
import scipy as sp

from eitprocessing.filters.mdn import MDNFilter

SAMPLE_FREQUENCY = 20
MINUTE = 60


def test_mdn_working():
    n_samples = 10000
    heart_rate = 70 / MINUTE
    respiratory_rate = 12 / MINUTE

    signal = np.random.default_rng().normal(2.0, 1.0, n_samples)

    mdn_filter = MDNFilter(sample_frequency=SAMPLE_FREQUENCY, heart_rate=heart_rate, respiratory_rate=respiratory_rate)
    filtered_signal = mdn_filter.apply_filter(signal)

    f, Pxx = sp.signal.welch(signal, fs=SAMPLE_FREQUENCY)  # noqa: N806
    _, filtered_Pxx = sp.signal.welch(filtered_signal, fs=SAMPLE_FREQUENCY)  # noqa: N806

    current_frequency_center = heart_rate
    while current_frequency_center < mdn_filter.noise_frequency_limit:
        notch_distance = mdn_filter.notch_distance
        frequency_band = (f > current_frequency_center - notch_distance) & (
            f < current_frequency_center + notch_distance
        )
        mean_power = np.mean(Pxx[frequency_band])

        filtered_mean_power = np.mean(filtered_Pxx[frequency_band])

        assert filtered_mean_power < mean_power

        current_frequency_center += heart_rate
