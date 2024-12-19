import itertools
from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class DetectRespiratoryHeartRates:
    """Detect te respiratory and heart rate from pixel data."""

    sample_frequency: float
    subject_type: str

    def apply(self, pixel_impedance: np.ndarray) -> tuple[float, float]:
        """Detect respiratory and heart rate based on pixel data."""
        nperseg = min(len(pixel_impedance), WELCH_WINDOW * self.sample_frequency)

        global_signal = np.nansum(pixel_impedance, axis=(1, 2))
        frequencies, global_power = signal.welch(
            global_signal - np.mean(global_signal),
            self.sample_frequency,
            nperseg=nperseg,
            noverlap=WELCH_OVERLAP * self.sample_frequency,
        )

        indices_respiratory_rate = np.nonzero(
            (frequencies > MINIMUM_RESPIRATORY_RATE[self.subject_type])
            & (frequencies < MAXIMUM_RESPIRATORY_RATE[self.subject_type])
        )
        indices_heart_rate = np.nonzero(
            (frequencies > MINIMUM_HEART_RATE[self.subject_type])
            & (frequencies < MAXIMUM_HEART_RATE[self.subject_type])
        )

        max_global_power = np.argmax(
            global_power[indices_respiratory_rate] == np.max(global_power[indices_respiratory_rate])
        )
        estimated_respiratory_rate: float = frequencies[indices_respiratory_rate][max_global_power]

        power_spectra = np.full((len(frequencies), NUM_ROWS, NUM_COLUMNS), np.nan)

        for row, column in itertools.product(range(NUM_ROWS), range(NUM_COLUMNS)):
            frequencies, power = signal.welch(
                signal.detrend(pixel_impedance[:, row, column], type="constant"),
                self.sample_frequency,
                nperseg=nperseg,
                noverlap=WELCH_OVERLAP * self.sample_frequency,
            )

            power_spectra[:, row, column] = power

        power_spectrum_normalizer = 1 / np.nansum(power_spectra, axis=0, keepdims=True)

        normalized_power_spectra = power_spectra * power_spectrum_normalizer
        summed_power_spectra = np.nansum(normalized_power_spectra, axis=(1, 2))

        summed_power_spectra_normalized = summed_power_spectra / np.sum(summed_power_spectra)
        global_power_normalized = global_power / np.sum(global_power)
        diff_global_summed_spectra = summed_power_spectra_normalized - global_power_normalized

        max_diff_index = np.argmax(
            diff_global_summed_spectra[indices_heart_rate] == np.max(diff_global_summed_spectra[indices_heart_rate])
        )
        estimated_heart_rate: float = frequencies[indices_heart_rate][max_diff_index]

        return estimated_respiratory_rate, estimated_heart_rate


MINUTE = 60

MINIMUM_HEART_RATE: dict[str, float] = {
    "adult": 40 / MINUTE,
    "neonate": 90 / MINUTE,
}
MAXIMUM_HEART_RATE: dict[str, float] = {
    "adult": 200 / MINUTE,
    "neonate": 210 / MINUTE,
}
MINIMUM_RESPIRATORY_RATE: dict[str, float] = {
    "adult": 6 / MINUTE,
    "neonate": 15 / MINUTE,
}
MAXIMUM_RESPIRATORY_RATE: dict[str, float] = {
    "adult": 60 / MINUTE,
    "neonate": 85 / MINUTE,
}

WELCH_OVERLAP: float = 30
WELCH_WINDOW: float = 60

NUM_ROWS = NUM_COLUMNS = 32
