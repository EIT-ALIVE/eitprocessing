from dataclasses import dataclass

import emd
import numpy as np

from eitprocessing.filters import TimeDomainFilter

N_BANDS = 5
REMOVE_BANDS = 3


@dataclass(kw_only=True)
class EMDFilter(TimeDomainFilter):
    """Empirical Mode Decomposition (EMD) filter for filtering in the time domain."""

    heart_rate: float
    sample_frequency: float
    n_bands: int = N_BANDS
    remove_bands: int = REMOVE_BANDS

    def apply_filter(self, input_data: np.ndarray) -> np.ndarray:
        """Generates EMD filtered version of original data."""
        limit = self.heart_rate / 0.67 / self.sample_frequency
        powers = np.arange(self.n_bands, 0, -1) - np.ceil(self.n_bands // 2)
        mask_frequencies = np.power(2, powers) * limit
        imfs = emd.sift.mask_sift(
            input_data,
            mask_freqs=mask_frequencies,
        )
        return input_data - imfs[:, : self.remove_bands].sum(axis=1)
