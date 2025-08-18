from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import feature as ski_feature
from skimage import segmentation as ski_segmentation

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.pixelmap import IntegerMap, PixelMap, TIVMap
from eitprocessing.features.pixel_breath import PixelBreath
from eitprocessing.parameters.tidal_impedance_variation import TIV
from eitprocessing.roi import PixelMask
from eitprocessing.utils import make_capture


@dataclass(kw_only=True, frozen=True)
class WatershedLungspace:
    """Create a pixel mask based on the watershed method.

    This method was designed to improve functional lung space detection in the presence of pendelluft. Functional lung
    space defined as pixels with a tidal impedance variation (TIV) at or above a percentage of the maximum TIV can
    result in underdetection of pixels that have reduced TIV due to the pendelluft phenomenon. An alternate approach
    using the pixel amplitude instead of the TIV results in overinclusion of pixels.

    In this method, the (inverse) watershed method is applied to the pixel amplitude. This results in distinct regions
    with high values for more central pixels, and low values for edge values. Regions where the highest value falls
    within the TIV-based functional lung space definition are included in the mask. Other regions are exluded. Pixels
    that fall outside the amplitude based functional lung space definiton are excluded.

    Example usage:
    ```python
    >>> mask = WatershedLungspace(threshold_fraction=0.15).apply(eit_data)
    >>> masked_eit_data = mask.apply(eit_data)
    ```

    Args:
        threshold_fraction (float):
            The fraction of the maximum TIV for the initial functional lung space definition that is used in the
            algorithm. Defaults to 0.15 (15%).
    """

    threshold_fraction: float = 0.15

    def __post_init__(self):
        if not isinstance(self.threshold_fraction, float):
            msg = "Threshold must be a float."
            raise TypeError(msg)
        if not (0 < self.threshold_fraction < 1):
            msg = "Threshold must be between 0 and 1."
            raise ValueError(msg)

    def apply(
        self, eit_data: EITData, *, timing_data: ContinuousData | None = None, captures: dict | None = None
    ) -> PixelMask:
        """Apply the watershed method to the EIT data.

        `BreathDetection` is used to find breaths in timing data. By default, the timing data is the summed pixel
        impedance. Alternative timing data, e.g., pressure data, can be provided.

        `TIV` is used to compute the mean tidal impedance variation (TIV) and mean pixel amplitude over the breaths. The
        absolute threshold is defined as 15% of the maximum TIV over all pixels. A functional TIV mask and functional
        amplitude mask are defined as all pixels with a TIV/amplitude of at least the absolute threshold.

        Local peaks in the amplitude map are found using `skimage.feature.peak_local_max`. These peaks are used to find
        the (inverse) watershed regions in the amplitude map. Regions whose peaks fall inside the functional TIV mask
        are included, the others are exluded. The final watershed mask is the intersection of the functional amplitude
        mask and the remaining included watershed regions.

        Watershed regions:
            The concept of watershed regions come from geography, where it refers to the area of land that drains into a
            single river or body of water. In the context of image processing, it refers to the region of an image that
            is associated with a particular local minimum. The region borders are defined by the 'ridges' between the
            local minima. The inverse watershed method uses maxima and 'valleys' between them. This algorithm uses the
            inverse method, where local maximum impedance values form the centers of watershed regions.

        Args:
            eit_data (EITData): The EIT data to apply the watershed method to.
            timing_data (ContinuousData | None):
                The timing data to use for the TIV calculation. If None, the summed impedance is used.
            captures (dict | None):
                An optional dictionary to capture intermediate results. If None, no captures are made.

        Returns:
            PixelMask: A mask of the functional lung space based on the watershed method.
        """
        capture = make_capture(captures)

        if timing_data is None:
            timing_data = eit_data.get_summed_impedance()

        tiv = TIV(pixel_breath=PixelBreath(phase_correction_mode="negative amplitude")).compute_pixel_parameter(
            eit_data=eit_data,
            continuous_data=timing_data,
            tiv_timing="continuous",
            sequence=None,
        )
        amplitude = TIV(pixel_breath=PixelBreath(phase_correction_mode="phase shift")).compute_pixel_parameter(
            eit_data=eit_data,
            continuous_data=timing_data,
            tiv_timing="pixel",
            sequence=None,
        )

        # The amplitude normally has less breaths; to prevent averaging over different breaths, only include breaths
        # that are in both sets
        included_breaths = (~np.all(np.isnan(tiv.values), axis=(1, 2))) & (
            ~np.all(np.isnan(amplitude.values), axis=(1, 2))
        )

        # Compute mean pixel TIV and associated threshold mask
        mean_tiv = TIVMap.from_aggregate(
            np.array(tiv.values)[included_breaths, :, :], np.nanmean, suppress_negative_warning=True
        )
        tiv_functional_mask = mean_tiv.create_mask_from_threshold(
            self.threshold_fraction, fraction_of_max=True, captures=(capture_tiv_mask := {})
        )
        capture("mean tiv", mean_tiv)
        capture("functional tiv mask", tiv_functional_mask)

        # Compute mean pixel amplitude and associated threshold mask
        mean_amplitude = TIVMap.from_aggregate(np.array(amplitude.values)[included_breaths, :, :], np.nanmean)
        amplitude_functional_mask = mean_amplitude.create_mask_from_threshold(
            capture_tiv_mask["actual threshold"], fraction_of_max=False
        )
        capture("mean amplitude", mean_amplitude)
        capture("functional amplitude mask", amplitude_functional_mask)

        # Find local peaks in amplitude map, and create map of peak locations. The watershed method needs numbered
        # markers that indicate numbered peaks (valleys, but the inverse), resulting in associated numbered regions.
        local_peaks = ski_feature.peak_local_max(
            mean_amplitude.to_non_nan_array(), exclude_border=False
        )  # xy-locations
        peaks_loc_bool = np.zeros(mean_amplitude.shape, dtype=bool)
        peaks_loc_bool[tuple(local_peaks.T)] = True  # boolean markers
        peaks_loc_int, _ = ndi.label(peaks_loc_bool)  # numbered markers
        marker_map = PixelMap(np.where(peaks_loc_int == 0, np.nan, peaks_loc_int))
        capture("local peaks", local_peaks)

        # Find markers that overlap with TIV mask
        masked_marker_map = tiv_functional_mask.apply(marker_map)
        markers_inside_tiv_mask = masked_marker_map.values[~np.isnan(masked_marker_map.values)]
        capture("included marker indices", markers_inside_tiv_mask)

        # Find the watershed regions
        watershed_regions = ski_segmentation.watershed(
            image=-mean_amplitude.values, markers=marker_map.to_integer_array()
        )

        # Only include regions whose' markers are inside the functional TIV mask. This combines the included regions
        # into a single region.
        included_region = PixelMask(np.isin(watershed_regions, markers_inside_tiv_mask))
        capture("included region", included_region)

        if captures is not None:
            capture("watershed regions", IntegerMap(watershed_regions))

            included_marker_indices = np.isin(peaks_loc_int, markers_inside_tiv_mask)
            included_peaks = np.argwhere(included_marker_indices)
            excluded_peaks = np.argwhere(~included_marker_indices)

            included_watershed_regions = np.where(included_region, watershed_regions, np.nan)

            capture("included peaks", included_peaks)
            capture("excluded peaks", excluded_peaks)
            capture("included watershed regions", included_watershed_regions)

        # Create the final mask, which only includes pixels that are inside the function amplitude mask, removing pixels
        # below the threshold.
        return included_region * amplitude_functional_mask
