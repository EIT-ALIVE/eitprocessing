import itertools
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from scipy import signal

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.intervaldata import IntervalData

MINUTE: Final = 60


@dataclass
class RateDetection:
    """Detect the respiratory and heart rate from EIT pixel data."""

    subject_type: Literal["adult", "neonate"]

    welch_window: float = 60
    welch_overlap: float = 30

    min_heart_rate: float | None = None
    max_heart_rate: float | None = None
    min_respiratory_rate: float | None = None
    max_respiratory_rate: float | None = None

    def __post_init__(self):
        for attr in (
            "min_heart_rate",
            "max_heart_rate",
            "min_respiratory_rate",
            "max_respiratory_rate",
        ):
            if getattr(self, attr) is None:
                setattr(self, attr, DEFAULT_SETTINGS[attr][self.subject_type])

    def detect_respiratory_heart_rate(self, eit_data: EITData) -> tuple[float, float]:
        """Detect respiratory and heart rate based on pixel data.

        This algorithm attempts
        """
        if eit_data.pixel_impedance is None:
            msg = "Can't detect heart rate without pixel data"
            raise NotImplementedError(msg)

        _, n_rows, n_cols = eit_data.pixel_impedance.shape
        pixel_impedance_detrended = signal.detrend(
            eit_data.pixel_impedance, type="constant", axis=0
        )
        summed_impedance = np.nansum(eit_data.pixel_impedance, axis=(1, 2))
        summed_impedance_detrended = signal.detrend(summed_impedance, type="constant")

        len_segment = min(
            len(pixel_impedance_detrended), self.welch_window * eit_data.sample_frequency
        )
        len_overlap = self.welch_overlap * eit_data.sample_frequency
        frequencies, total_power = signal.welch(
            summed_impedance_detrended,
            eit_data.sample_frequency,
            nperseg=len_segment,
            noverlap=len_overlap,
        )

        indices_respiratory_rate = np.nonzero(
            (frequencies >= self.min_respiratory_rate)
            & (frequencies <= self.max_respiratory_rate),
        )
        indices_heart_rate = np.nonzero(
            (frequencies >= self.min_heart_rate) & (frequencies <= self.max_heart_rate),
        )

        max_total_power = np.argmax(
            total_power[indices_respiratory_rate] == np.max(total_power[indices_respiratory_rate]),
        )
        estimated_respiratory_rate: float = frequencies[indices_respiratory_rate][max_total_power]

        power_spectra = np.full((len(frequencies), n_rows, n_cols), np.nan)

        included_pixels = (np.std(pixel_impedance_detrended, axis=0) > 0) & np.all(
            (~np.isnan(pixel_impedance_detrended)), axis=0
        )

        for row, column in itertools.product(range(n_rows), range(n_cols)):
            pixel_data = pixel_impedance_detrended[:, row, column]

            if not included_pixels[row, column]:
                continue

            pixel_data_detrended = signal.detrend(pixel_data, type="constant")

            frequencies, power = signal.welch(
                pixel_data_detrended,
                eit_data.sample_frequency,
                nperseg=len_segment,
                noverlap=len_overlap,
            )
            power_spectra[:, row, column] = power

        power_spectrum_normalizer = np.full(power_spectra.shape, np.nan)
        power_spectrum_normalizer[:, included_pixels] = 1 / np.nansum(
            power_spectra[:, included_pixels], axis=0, keepdims=True
        )

        normalized_power_spectra = power_spectra * power_spectrum_normalizer
        summed_power_spectra = np.nansum(normalized_power_spectra, axis=(1, 2))

        summed_power_spectra_normalized = summed_power_spectra / np.sum(
            summed_power_spectra,
        )
        total_power_normalized = total_power / np.sum(total_power)
        diff_total_summed_spectra = summed_power_spectra_normalized - total_power_normalized

        max_diff_index = np.argmax(
            diff_total_summed_spectra[indices_heart_rate]
            == np.max(diff_total_summed_spectra[indices_heart_rate]),
        )
        estimated_heart_rate: float = frequencies[indices_heart_rate][max_diff_index]

        interval = eit_data.time[0], eit_data.time[-1]
        rr_data = IntervalData(
            "respiratory_rate",
            "Estimated respiratory rate as determined by RateDetection",
            unit="Hz",
            category="respiratory rate",
            intervals=[interval],
            values=[estimated_respiratory_rate],
            parameters=dict(vars(self)),
            derived_from=[eit_data],
            default_partial_inclusion=True,
        )

        hr_data = IntervalData(
            "heart_rate",
            "Estimated heart rate as determined by RateDetection",
            unit="Hz",
            category="heart rate",
            intervals=[interval],
            values=[estimated_heart_rate],
            parameters=dict(vars(self)),
            derived_from=[eit_data],
            default_partial_inclusion=True,
        )

        return rr_data, hr_data


DEFAULT_SETTINGS = {
    "min_heart_rate": {"adult": 40 / MINUTE, "neonate": 90 / MINUTE},
    "max_heart_rate": {"adult": 200 / MINUTE, "neonate": 210 / MINUTE},
    "min_respiratory_rate": {"adult": 6 / MINUTE, "neonate": 15 / MINUTE},
    "max_respiratory_rate": {"adult": 60 / MINUTE, "neonate": 85 / MINUTE},
}
