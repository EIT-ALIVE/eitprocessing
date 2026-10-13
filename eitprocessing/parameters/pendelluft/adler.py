import itertools
import warnings

import numpy as np

from eitprocessing.datahandling.breath import Breath
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.pixelmap import PendelluftMap, PixelMap
from eitprocessing.features import breath_detection
from eitprocessing.features.pixel_breath import PixelBreath
from eitprocessing.parameters.pendelluft import PendelluftResult
from eitprocessing.utils import make_capture

MIN_MEAN_PENDELLUFT_VALUE = 0
MAX_MEAN_PENDELLUFT_VALUE = 0.15


class PendelluftAdler:
    """Pendelluft calculation following Adler et al.

    Implements the pendelluft calculation method described in Adler et al. on pendelluft:
    https://doi.org/10.1088/1361-6579/ad7fca
    """

    def apply(
        self, eit_data: EITData, captures: dict | None = None, early_pendelluft_percentile: float = 5.0
    ) -> PendelluftMap:
        """Calculate pendelluft (FRIC) following Adler et al.

        Steps:
        - Differentiate pixel impedance curves.
        - Detect breaths using BreathDetection and PixelBreath.
        - Compute positive and negative tidal impedance variation (TIV).
        - Calculate breathwise and pixelwise FRIC.

        Args:
            sequence (Sequence): Data container with EIT signals.
            preprocessing_method (Literal["emc", "adler"]): Preprocessing method used.
            captures (dict): Dictionary to store captures for visualization/debugging.

        Returns:
            pendelluft_adler_mean (float): Mean pendelluft value (FRIC_mean).
        """
        capture = make_capture(captures)

        pixel_impedance = eit_data.pixel_impedance
        summed_impedance = eit_data.get_summed_impedance()
        frame_shape = pixel_impedance.shape[1:]

        # Differentiate pixel impedance over time
        pixel_impedance_derivative = np.diff(pixel_impedance, axis=0)

        bd = breath_detection.BreathDetection()
        breaths = bd.find_breaths(summed_impedance)
        breath_values = breaths.values or []

        start_times = [b.start_time for b in breath_values]
        start_global_inspiration_indices = np.searchsorted(eit_data.time, start_times)
        middle_times = [b.middle_time for b in breath_values]
        end_times = [b.end_time for b in breath_values]

        # Detect pixelwise breath timings
        pb = PixelBreath(phase_correction_mode="phase shift")
        pixel_breaths = pb.find_pixel_breaths(
            eit_data,
            summed_impedance,
            result_label="pixel_breaths_adler",
        )
        values = np.asarray(pixel_breaths.values, dtype=object)

        # Split start / middle indices
        pixel_start_indices, pixel_middle_indices = self._pixel_breath_indices(values, eit_data.time)

        # Logic for tstart shift if pixel decreases before global start.
        t_start_indices = self._breathwise_start_indices(
            pixel_start_indices, start_global_inspiration_indices, percentile=early_pendelluft_percentile
        )

        # Save invalid masks before int conversion: astype(int) turns NaN into garbage
        # (typically INT64_MIN), which would silently overflow on subtraction.
        t_start_invalid = np.isnan(t_start_indices[1:-1])  # (num_breaths,)
        pixel_middle_invalid = np.isnan(pixel_middle_indices[1:-1])  # (num_breaths, rows, cols)

        # Remove first and last breaths; NaN entries are guarded by the saved masks above.
        with np.errstate(invalid="ignore"):
            t_start_indices = t_start_indices[1:-1].astype(int)
            pixel_middle_indices = pixel_middle_indices[1:-1].astype(int)

        valid_pixel_mask = ~np.isnan(pixel_impedance).all(axis=0)  # (rows, cols)

        tiv_pos_per_pixel_per_breath, tiv_neg_per_pixel_per_breath = self._compute_tiv(
            pixel_impedance_derivative,
            t_start_indices,
            pixel_middle_indices,
            t_start_invalid,
            pixel_middle_invalid,
            valid_pixel_mask,
            frame_shape,
        )

        # Calculate FRIC_n per breath: TIV_negative_n / (TIV_positive_n + TIV_negative_n)
        fric_per_pixel_per_breath = tiv_neg_per_pixel_per_breath / (
            tiv_pos_per_pixel_per_breath + tiv_neg_per_pixel_per_breath
        )

        tiv_pos_per_breath = np.nansum(tiv_pos_per_pixel_per_breath, axis=(1, 2))
        tiv_neg_per_breath = np.nansum(tiv_neg_per_pixel_per_breath, axis=(1, 2))
        fric_per_breath = tiv_neg_per_breath / (tiv_pos_per_breath + tiv_neg_per_breath)

        tiv_pos_per_pixel = np.nanmean(tiv_pos_per_pixel_per_breath, axis=0)
        tiv_neg_per_pixel = np.nanmean(tiv_neg_per_pixel_per_breath, axis=0)
        fric_per_pixel = tiv_neg_per_pixel / (tiv_pos_per_pixel + tiv_neg_per_pixel)

        # PS: we should think about a good system for storing intermediate
        # results. Maybe something similar how we use capture for plotting? Make
        # an IO-specific class that handles saving results?

        # Store captures for visualization/debugging
        capture("filtered_global", summed_impedance)
        capture("start_global_inspiration_indices", start_global_inspiration_indices)
        capture("eit_data", eit_data)
        capture("pixel_impedance_derivative", pixel_impedance_derivative)
        capture("quarter_index", len(summed_impedance.time) // 4)
        capture("FRIC_i", fric_per_pixel)

        # Generate warnings for implausible results
        pendelluft_mean = np.nanmean(fric_per_breath)
        if pendelluft_mean < MIN_MEAN_PENDELLUFT_VALUE:
            warnings.warn("Pendelluft value is negative. Likely incorrect calculation.")
        elif pendelluft_mean > MAX_MEAN_PENDELLUFT_VALUE:
            warnings.warn("Pendelluft value is high. Likely incorrect calculation.")

        return PendelluftResult(
            mean_value=np.nanmean(fric_per_breath),
            breath_values=fric_per_breath,
            mean_map=PendelluftMap(fric_per_pixel),
            breath_maps=tuple(PendelluftMap(m) for m in fric_per_pixel_per_breath),
        )

    def _pixel_breath_indices(
        self,
        values: np.ndarray,
        time: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract per-pixel breath start and middle (end-of-inspiration) frame indices.

        Args:
            values (np.ndarray): (n_breaths, rows, cols) object array of Breath objects.
            time (np.ndarray): 1D time array for the regional signal.

        Returns:
            tuple: (start_indices, middle_indices), each (n_breaths, rows, cols) float,
                NaN where no valid Breath object was found.
        """
        # Flatten to 1D so we can use a list comprehension instead of ndenumerate.
        # ndenumerate yields (multi-index, value) pairs and requires indexed assignment
        # into a 4D array — list comprehension over the flat array is noticeably faster
        # for typical sizes (~20 breaths × 32 × 32 pixels = ~20 000 iterations).
        flat = values.ravel()
        start_times = np.array(
            [b.start_time if isinstance(b, Breath) else np.nan for b in flat],
            dtype=float,
        ).reshape(values.shape)
        middle_times = np.array(
            [b.middle_time if isinstance(b, Breath) else np.nan for b in flat],
            dtype=float,
        ).reshape(values.shape)

        start_indices = np.searchsorted(time, start_times).astype(float)
        middle_indices = np.searchsorted(time, middle_times).astype(float)
        start_indices[np.isnan(start_times)] = np.nan
        middle_indices[np.isnan(middle_times)] = np.nan

        return start_indices, middle_indices

    def _compute_tiv(
        self,
        derivative: np.ndarray,
        t_start_indices: np.ndarray,
        pixel_middle_indices: np.ndarray,
        t_start_invalid: np.ndarray,
        pixel_middle_invalid: np.ndarray,
        valid_pixel_mask: np.ndarray,
        frame_shape: tuple,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-pixel per-breath positive and negative TIV via cumulative sums.

        Precomputes cumulative sums of the positive and negative derivative components
        once over the full time axis; each breath is then two O(rows×cols) indexed
        lookups rather than a boolean window mask (~50–150 MB per breath).

        Args:
            derivative (np.ndarray): (T-1, rows, cols) frame-to-frame impedance differences.
            t_start_indices (np.ndarray): (num_breaths,) per-breath start frame indices.
            pixel_middle_indices (np.ndarray): (num_breaths, rows, cols) per-pixel end indices.
            t_start_invalid (np.ndarray): (num_breaths,) boolean mask for invalid start indices.
            pixel_middle_invalid (np.ndarray): (num_breaths, rows, cols) mask for invalid end indices.
            valid_pixel_mask (np.ndarray): (rows, cols) True for pixels that are not all-NaN.
            frame_shape (tuple): (rows, cols).

        Returns:
            tuple: (tiv_pos_per_pixel_per_breath, tiv_neg_per_pixel_per_breath),
                each (num_breaths, rows, cols), NaN for invalid entries.
        """
        # Prepend a zero-row so index t maps to "sum of first t elements" (exclusive end).
        zero = np.zeros((1, *frame_shape))
        pos_cumsum = np.concatenate([zero, np.cumsum(np.where(derivative > 0, derivative, 0.0), axis=0)], axis=0)
        neg_cumsum = np.concatenate([zero, np.cumsum(np.where(derivative < 0, -derivative, 0.0), axis=0)], axis=0)

        num_breaths = len(t_start_indices)
        row_idx, col_idx = np.mgrid[0 : frame_shape[0], 0 : frame_shape[1]]  # (rows, cols) for fancy indexing
        n_frames = pos_cumsum.shape[0]

        tiv_pos_per_pixel_per_breath = np.full((num_breaths, *frame_shape), np.nan)
        tiv_neg_per_pixel_per_breath = np.full((num_breaths, *frame_shape), np.nan)

        for b_idx, t_start in enumerate(t_start_indices):
            if t_start_invalid[b_idx]:
                continue

            t_end_2d = pixel_middle_indices[b_idx]  # (rows, cols)
            invalid = pixel_middle_invalid[b_idx] | (t_end_2d <= t_start)  # (rows, cols)
            t_end_clamped = np.clip(np.where(invalid, t_start, t_end_2d), 0, n_frames - 1)

            pos_sum = pos_cumsum[t_end_clamped, row_idx, col_idx] - pos_cumsum[t_start, row_idx, col_idx]
            neg_sum = neg_cumsum[t_end_clamped, row_idx, col_idx] - neg_cumsum[t_start, row_idx, col_idx]

            has_window = valid_pixel_mask & ~invalid & (t_end_2d > t_start)
            tiv_pos_per_pixel_per_breath[b_idx] = np.where(has_window, pos_sum, np.nan)
            tiv_neg_per_pixel_per_breath[b_idx] = np.where(has_window, neg_sum, np.nan)

        return tiv_pos_per_pixel_per_breath, tiv_neg_per_pixel_per_breath

    def _breathwise_start_indices(
        self,
        pixel_start_indices: np.ndarray,
        start_global_inspiration_indices: np.ndarray,
        percentile: float = 5.0,
    ) -> np.ndarray:
        """Compute per-breath start index from the earliest-starting fraction of pixels.

        Takes the given percentile of per-pixel start indices across the frame, then
        clips to the global inspiration start so t_start never exceeds the global onset.

        Args:
            pixel_start_indices (np.ndarray): (n_breaths, rows, cols) pixel start indices.
            start_global_inspiration_indices (np.ndarray): (n_breaths,) global start indices.
            percentile (float): Percentile of pixel start indices to use (0–100). Default = 5.0.

        Returns:
            np.ndarray: (n_breaths,) per-breath start indices (float, NaN if no active pixels).
        """
        start_indices = np.nanpercentile(
            pixel_start_indices,
            percentile,
            axis=(1, 2),
            method="higher",
        )
        return np.clip(start_indices, None, start_global_inspiration_indices)
