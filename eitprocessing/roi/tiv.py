import sys
from dataclasses import dataclass
from typing import Literal

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
    """Create a pixel mask by thresholding the mean TIV or amplitude.

    This defines the functional lung space as all pixels with a tidal impedance variation (TIV, default) or amplitude of
    at least the provided fractional threshold of the maximum TIV or amplitude.

    Warning:
        The amplitude mode is not recommended for functional lung space detection, as it potentially includes
        reconstruction artifacts. The option is provided for completeness and use in other algorithms, namely
        WatershedLungspace.

    Example usage:
    ```python
    >>> mask = TIVLungspace(threshold=0.15, mode="TIV").apply(eit_data)
    >>> masked_eit_data = mask.apply(eit_data)
    ```

    Args:
        threshold (float):
            The fraction of the maximum TIV or amplitude that is used as threshold. Defaults to 0.15 (15%).
        mode (Literal["TIV", "amplitude"]): Whether to use TIV (default) or amplitude for the mask definition.

    """

    threshold: float = 0.15
    mode: Literal["TIV", "amplitude"] = "TIV"

    def __post_init__(self):
        if not isinstance(self.threshold, float):
            msg = "Threshold must be a float."
            raise TypeError(msg)
        if not (0 < self.threshold < 1):
            msg = "Threshold must be between 0 and 1."
            raise ValueError(msg)

        if self.mode not in ("TIV", "amplitude"):
            msg = f"Unknown mode '{self.mode}'. Supported modes are 'TIV' and 'amplitude'."
            raise ValueError(msg)

    def apply(
        self, eit_data: EITData, *, timing_data: ContinuousData | None = None, captures: dict | None = None
    ) -> PixelMask:
        """Apply the TIV or amplitude thresholding to the EIT data.

        `BreathDetection` is used to find breaths in timing data. By default, the timing data is the summed pixel
        impedance. Alternative timing data, e.g., pressure data, can be provided.

        Then, `TIV` is used to compute the mean tidal impedance variation (TIV) or mean pixel amplitude over the
        breaths.

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

        if self.mode == "TIV":
            magnitude = TIV(
                pixel_breath=PixelBreath(phase_correction_mode="negative amplitude")
            ).compute_pixel_parameter(
                eit_data=eit_data,
                continuous_data=timing_data,
                tiv_timing="continuous",
                sequence=None,
            )
        else:
            magnitude = TIV(pixel_breath=PixelBreath(phase_correction_mode="phase shift")).compute_pixel_parameter(
                eit_data=eit_data,
                continuous_data=timing_data,
                tiv_timing="pixel",
                sequence=None,
            )

        if not len(magnitude):
            msg = f"No breaths were detected. Cannot compute {self.mode}"
            raise ValueError(msg)

        if np.all(np.isnan(magnitude.values)):
            msg = f"No non-nan {self.mode} were found."
            exc = ValueError(msg)
            if sys.version_info >= (3, 11):
                exc.add_note("This may be due to too few breaths being detected in the data.")
            raise exc

        capture(f"{self.mode} values", magnitude)

        mean_magnitude = TIVMap.from_aggregate(np.array(magnitude.values), np.nanmean, suppress_negative_warning=True)
        capture(f"mean {self.mode}", mean_magnitude)

        return mean_magnitude.create_mask_from_threshold(self.threshold, fraction_of_max=True)
