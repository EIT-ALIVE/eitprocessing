import itertools
from dataclasses import dataclass
from enum import auto
from typing import TYPE_CHECKING, Final

import numpy as np
from scipy import signal
from strenum import LowercaseStrEnum

from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.sequence import Sequence

if TYPE_CHECKING:
    from eitprocessing.datahandling.eitdata import EITData

MINUTE: Final = 60
NUM_ROWS = NUM_COLUMNS = 32


class SubjectType(LowercaseStrEnum):
    """Type of subject the data was gathered from."""

    ADULT = auto()
    NEONATE = auto()


@dataclass
class RateDetection:
    """Detect the respiratory and heart rate from EIT pixel data."""

    subject_type: SubjectType | str

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

    def detect_respiratory_heart_rate(self, sequence: Sequence, eitdata_label: str) -> tuple[float, float]:
        """Detect respiratory and heart rate based on pixel data.

        This algorithm attempts
        """
        eitdata: EITData = sequence.eit_data[eitdata_label]
        if eitdata.pixel_impedance is None:
            msg = "Can't detect heart rate without pixel data"
            raise NotImplementedError(msg)

        data_without_mean = eitdata.pixel_impedance - np.mean(eitdata.pixel_impedance, axis=0)

        nperseg = min(len(data_without_mean), self.welch_window * eitdata.framerate)
        noverlap = self.welch_overlap * eitdata.framerate
        total_signal = np.nansum(data_without_mean, axis=(1, 2))
        frequencies, total_power = signal.welch(
            total_signal - np.mean(total_signal),
            eitdata.framerate,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        indices_respiratory_rate = np.nonzero(
            (frequencies >= self.min_respiratory_rate) & (frequencies <= self.max_respiratory_rate),
        )
        indices_heart_rate = np.nonzero(
            (frequencies >= self.min_heart_rate) & (frequencies <= self.max_heart_rate),
        )

        max_total_power = np.argmax(
            total_power[indices_respiratory_rate] == np.max(total_power[indices_respiratory_rate]),
        )
        estimated_respiratory_rate: float = frequencies[indices_respiratory_rate][max_total_power]

        power_spectra = np.full((len(frequencies), NUM_ROWS, NUM_COLUMNS), np.nan)

        for row, column in itertools.product(range(NUM_ROWS), range(NUM_COLUMNS)):
            pixel_data = data_without_mean[:, row, column]
            frequencies, power = signal.welch(
                pixel_data - np.mean(pixel_data),
                eitdata.framerate,
                nperseg=nperseg,
                noverlap=noverlap,
            )

            power_spectra[:, row, column] = power

        power_spectrum_normalizer = 1 / np.nansum(power_spectra, axis=0, keepdims=True)

        normalized_power_spectra = power_spectra * power_spectrum_normalizer
        summed_power_spectra = np.nansum(normalized_power_spectra, axis=(1, 2))

        summed_power_spectra_normalized = summed_power_spectra / np.sum(
            summed_power_spectra,
        )
        total_power_normalized = total_power / np.sum(total_power)
        diff_total_summed_spectra = summed_power_spectra_normalized - total_power_normalized

        max_diff_index = np.argmax(
            diff_total_summed_spectra[indices_heart_rate] == np.max(diff_total_summed_spectra[indices_heart_rate]),
        )
        estimated_heart_rate: float = frequencies[indices_heart_rate][max_diff_index]

        sequence.interval_data.add(
            IntervalData(
                "respiratory_rate",
                "Estimated respiratory rate as determined by RateDetection",
                unit="Hz",
                category="respiratory rate",
                intervals=[(eitdata.time[0], eitdata.time[-1])],
                values=[estimated_respiratory_rate],
                parameters=dict(vars(self)),
                derived_from=[eitdata],
                default_partial_inclusion=True,
            ),
        )

        sequence.interval_data.add(
            IntervalData(
                "heart_rate",
                "Estimated heart rate as determined by RateDetection",
                unit="Hz",
                category="heart rate",
                intervals=[(eitdata.time[0], eitdata.time[-1])],
                values=[estimated_heart_rate],
                parameters=dict(vars(self)),
                derived_from=[eitdata],
                default_partial_inclusion=True,
            ),
        )

        return estimated_respiratory_rate, estimated_heart_rate


DEFAULT_SETTINGS = {
    "min_heart_rate": {
        SubjectType.ADULT: 40 / MINUTE,
        SubjectType.NEONATE: 90 / MINUTE,
    },
    "max_heart_rate": {
        SubjectType.ADULT: 200 / MINUTE,
        SubjectType.NEONATE: 210 / MINUTE,
    },
    "min_respiratory_rate": {
        SubjectType.ADULT: 6 / MINUTE,
        SubjectType.NEONATE: 15 / MINUTE,
    },
    "max_respiratory_rate": {
        SubjectType.ADULT: 60 / MINUTE,
        SubjectType.NEONATE: 85 / MINUTE,
    },
}
