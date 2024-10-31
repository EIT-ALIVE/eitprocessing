import warnings
from dataclasses import dataclass

import numpy as np
from scipy import signal

from eitprocessing.filters import TimeDomainFilter

MINUTE = 60
NOISE_FREQUENCY_LIMIT: float = 220 / MINUTE
NOTCH_DISTANCE: float = 10 / MINUTE

UPPER_RESPIRATORY_RATE_LIMIT: float = 2
UPPER_HEART_RATE_LIMIT: float = 10


@dataclass
class MDNFilter(TimeDomainFilter):
    """Multiple Digital Notch filter.

    This filter is used to remove heart rate noise from data. It uses multiple
    notch filters to remove the frequency surrounding the heart rate (± the
    notch distance) and its harmonics, up to the Nyquist frequency. The filter
    also applies a high pass filter above the noise frequency limit.

    NB: the respiratory and heart rate should be in provided Hz, not BPM.

    Args:
      sample_frequency: the sample frequency of the data in Hz
      respiratory_rate: the respiratory rate of the subject in Hz
      heart_rate: the heart rate of the subject in Hz
      noise_frequency_limit: the highest frequency to filter in Hz
      notch_distance: the distance from the heart rate to filter in Hz

    """

    sample_frequency: float
    respiratory_rate: float
    heart_rate: float
    noise_frequency_limit: float = NOISE_FREQUENCY_LIMIT
    notch_distance: float = NOTCH_DISTANCE

    def __post_init__(self):
        if self.respiratory_rate > UPPER_RESPIRATORY_RATE_LIMIT:
            msg = (
                f"The provided respiratory rate ({self.respiratory_rate:.1f}) "
                "is higher than 2 Hz (120 BPM). "
                "Make sure this is correct, and to use the correct unit."
            )
            warnings.warn(msg)

        if self.heart_rate > UPPER_HEART_RATE_LIMIT:
            msg = (
                f"The provided heart rate ({self.heart_rate:.1f}) is higher "
                "than 10 Hz (600 BPM). "
                "Make sure this is correct, and to use the correct unit."
            )
            warnings.warn(msg)

    def apply_filter(self, input_data: np.ndarray, axis: int = -1) -> np.ndarray:
        """Filter data using multiple multiple digital notch filters."""

        class FrequencyLimitReached(Exception):  # noqa: N818
            """The highest frequency is reached."""

        def filter_(data: np.ndarray, harmonic: int) -> np.ndarray:
            lower_limit = self.heart_rate * harmonic - self.notch_distance
            upper_limit = self.heart_rate * harmonic + self.notch_distance

            if lower_limit >= self.noise_frequency_limit * 2:
                msg = """Frequency band (partly) lies above twice the noise
                frequency limit."""
                raise FrequencyLimitReached(msg)

            upper_limit = min(self.noise_frequency_limit * 2, upper_limit)

            if harmonic == 1:
                new_lower_limit = (self.heart_rate + self.respiratory_rate) / 2
                lower_limit = max(lower_limit, new_lower_limit)

            sos = signal.butter(
                10,
                [lower_limit, upper_limit],
                fs=self.sample_frequency,
                btype="bandstop",
                output="sos",
            )

            return signal.sosfiltfilt(sos, data, axis=axis)

        harmonic = 1
        filtered_data = filter_(input_data, harmonic)
        try:
            while True:
                harmonic += 1
                filtered_data = filter_(filtered_data, harmonic)
        except FrequencyLimitReached:
            pass

        # Filtere everything above noise limit
        sos = signal.butter(
            5,
            self.noise_frequency_limit,
            fs=self.sample_frequency,
            btype="low",
            output="sos",
        )
        return signal.sosfiltfilt(sos, filtered_data, axis)
