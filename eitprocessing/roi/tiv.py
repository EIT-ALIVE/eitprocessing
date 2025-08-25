import sys
from dataclasses import dataclass

import numpy as np

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.pixelmap import TIVMap
from eitprocessing.features.pixel_breath import PixelBreath
from eitprocessing.parameters.tidal_impedance_variation import TIV
from eitprocessing.roi import PixelMask
from eitprocessing.utils import make_capture


@dataclass(kw_only=True, frozen=True)
class TIVLungspace:
    """Create a pixel mask by thresholding the mean TIV.

    This defines the functional lung space as all pixels with a tidal impedance variation (TIV) of at least the provided
    fractional threshold of the maximum TIV.

    Example usage:
    ```python
    >>> mask = TIVLungspace(threshold=0.15).apply(eit_data)
    >>> masked_eit_data = mask.apply(eit_data)
    ```

    Args:
        threshold (float): The fraction of the maximum TIV that is used as threshold. Defaults to 0.15 (15%).
    """

    threshold: float = 0.15

    def __post_init__(self):
        if not isinstance(self.threshold, float):
            msg = "Threshold must be a float."
            raise TypeError(msg)
        if not (0 < self.threshold < 1):
            msg = "Threshold must be between 0 and 1."
            raise ValueError(msg)

    def apply(
        self, eit_data: EITData, *, timing_data: ContinuousData | None = None, captures: dict | None = None
    ) -> PixelMask:
        """Apply the TIV thresholding to the EIT data.

        `BreathDetection` is used to find breaths in timing data. By default, the timing data is the summed pixel
        impedance. Alternative timing data, e.g., pressure data, can be provided.

        Then, `TIV` is used to compute the TIV for each breath. The mean TIV over all breaths is computed,
        and pixels with a mean TIV above the threshold are included in the mask.

        Args:
            eit_data (EITData): The EIT data to process.
            timing_data (ContinuousData | None, optional):
                Optionally, alternative continuous data to use for breath detection. If `None`, the summed pixel
                impedance is used. Defaults to `None`.
            captures (dict | None, optional):
                A dictionary to store intermediate results. If `None`, no intermediate results are stored. Defaults to
                `None`.
        """
        capture = make_capture(captures)

        if timing_data is None:
            timing_data = eit_data.get_summed_impedance()

        tiv_per_breath = TIV(
            pixel_breath=PixelBreath(phase_correction_mode="negative amplitude")
        ).compute_pixel_parameter(
            eit_data=eit_data,
            continuous_data=timing_data,
            tiv_timing="continuous",
            sequence=None,
        )
        if not len(tiv_per_breath):
            msg = "No breaths were detected. Cannot compute TIV."
            raise ValueError(msg)

        if np.all(np.isnan(tiv_per_breath.values)):
            # not able to write tests to capture this case, but included for safety
            msg = "No non-nan TIV values were found."
            exc = ValueError(msg)
            if sys.version_info >= (3, 11):
                exc.add_note("This may be due to too few breaths being detected in the data.")
            raise exc

        capture("TIV per breath", tiv_per_breath)

        mean_tiv = TIVMap.from_aggregate(np.array(tiv_per_breath.values), np.nanmean, suppress_negative_warning=True)
        capture("mean TIV", mean_tiv)

        return mean_tiv.create_mask_from_threshold(self.threshold, fraction_of_max=True)
