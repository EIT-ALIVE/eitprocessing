import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from scipy import signal

from eitprocessing.datahandling.eitdata import EITData

if TYPE_CHECKING:
    from eitprocessing.plotting.rate_detection import RateDetectionPlotting

MINUTE: Final = 60
MIN_WELCH_WINDOW_LENGTH: Final = 10


@dataclass(frozen=True)
class RateDetection:
    """Detect the respiratory and heart rate from EIT pixel data.

    This algorithm attempts to detect the respiratory and heart rate from EIT pixel data. It is based on the observation
    that many high-amplitude pixels have the respiratory rate as the main frequency, while in low-amplitude pixels the
    power of the heart rate is relatively high. The algorithm uses Welch's method to estimate the power spectrum of the
    summed pixel data and individual pixels. It then identifies the respiratory rate as the frequency with the highest
    power for the summed pixels within the specified range. The power spectra of the individual pixels are normalized
    and averaged. The normalized power spectrum of the summed pixels is subtracted from the average of the normalized
    individual power spectra. The frequency with the highest relative power in this difference within the specified
    range is taken as the heart rate.

    If either rate is variable, the algorithm will in most cases return an average frequency. If there are multiple
    distinct frequencies, e.g., due to a change in the controlled respiratory rate, multiple peaks might be visible in
    the power spectrum. The algorithm will only return the frequency with the highest power in the specified range.

    The algorithm can't distinguish between the respiratory and heart rate if they are too close together, especially in
    very short signals (or when the Welch window is short). The algorithm can distinguish between both rates if the
    heart rate is at one of the harmonics of the respiratory rate.

    If the `refine_estimated_frequency` attribute is set to False, the estimated frequency is simply the location of the
    peak of the power Welch spectrum. Since Welch's method results in a limited number of frequency bins, this can lead
    to inaccuracies, especially with short or low-sample frequency data. If `refine_estimated_frequency` is set to True
    (default), the estimated frequency is refined using parabolic interpolation, which often yields more accurate
    results, even with short signals, low signal-to-noise ratio and very similar (but distinct) respiratory and heart
    rates. See e.g. [Quadratic Interpolation of Spectral
    Peaks](https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html).

    The respiratory and heart rate limits can be set when initializing this algorithm. Default values for adults and
    neonates are set in `DEFAULT_RATE_LIMITS`.

    Although the algorithm might perform reasonably on data as short as 10 seconds (200 samples), it is recommended to
    use at least 30 seconds of data for reliable results. Longer data wil yield more reliable results, but only if the
    respiratory and heart rates are stable.

    Note that the algorithm performs best if no large changes in end-expiratory impedance occur in the data, the data is
    unfiltered, and the respiratory and heart rates are relatively stable.

    Attributes:
        subject_type: The type of subject, either "adult" or "neonate". This affects the default settings for the
            minimum and maximum heart and respiratory rates.
        welch_window: The length of the Welch window in seconds.
        welch_overlap: The fraction overlap between Welch windows (e.g., 0.5 = 50% overlap).
        min_heart_rate: The minimum heart rate in Hz. If None, the default value for the subject type is used.
        max_heart_rate: The maximum heart rate in Hz. If None, the default value for the subject type is used.
        min_respiratory_rate:
            The minimum respiratory rate in Hz. If None, the default value for the subject type is used.
        max_respiratory_rate:
            The maximum respiratory rate in Hz. If None, the default value for the subject type is used.
        refine_estimated_frequency:
            If True, the estimated frequency is refined using parabolic interpolation. If False, the frequency with the
            highest power is used as the estimated frequency.
    """

    subject_type: Literal["adult", "neonate"]

    min_heart_rate: float
    max_heart_rate: float
    min_respiratory_rate: float
    max_respiratory_rate: float

    welch_window: float = 30.0
    welch_overlap: float = 0.5

    refine_estimated_frequency: bool = True

    def __init__(
        self,
        subject_type: Literal["adult", "neonate"],
        *,
        welch_window: float = 30.0,
        welch_overlap: float = 0.5,
        min_heart_rate: float | None = None,
        max_heart_rate: float | None = None,
        min_respiratory_rate: float | None = None,
        max_respiratory_rate: float | None = None,
        refine_estimated_frequency: bool = True,
    ):
        if welch_overlap >= 1:
            msg = "Welch overlap must be less than 1.0 (100%)."
            raise ValueError(msg)
        if welch_overlap < 0:
            msg = "Welch overlap must be at least 0 (0%)."
            raise ValueError(msg)

        if welch_window < MIN_WELCH_WINDOW_LENGTH:
            msg = "Welch window must be at least 10 seconds."
            raise ValueError(msg)

        if subject_type not in ("adult", "neonate"):
            msg = f"Invalid subject type: {subject_type}. Must be 'adult' or 'neonate'."
            raise ValueError(msg)

        object.__setattr__(self, "subject_type", subject_type)
        object.__setattr__(self, "welch_window", welch_window)
        object.__setattr__(self, "welch_overlap", welch_overlap)
        object.__setattr__(self, "refine_estimated_frequency", refine_estimated_frequency)

        for attr in (
            "min_heart_rate",
            "max_heart_rate",
            "min_respiratory_rate",
            "max_respiratory_rate",
        ):
            object.__setattr__(self, attr, locals().get(attr, None) or DEFAULT_RATE_LIMITS[attr][self.subject_type])

    def apply(
        self,
        eit_data: EITData,
        *,
        captures: dict | None = None,
        suppress_length_warnings: bool = False,
        suppress_edge_case_warning: bool = False,
    ) -> tuple[float, float]:
        """Detect respiratory and heart rate based on pixel data.

        NB: the respiratory and heart rate are returned in Hz. Multiply by 60 to convert to breaths/beats per minute.

        Arguments:
            eit_data: EITData object containing pixel impedance data and sample frequency.
            captures:
                Optional dictionary to capture additional information during processing. Can be used for plotting or
                debugging purposes.
            suppress_length_warnings:
                If True, suppress warnings about segment length being larger than the data length or overlap being
                larger than segment length. Defaults to False.
            suppress_edge_case_warning:
                If True, suppress warnings about the maximum power being at the edge of the frequency range, which
                prevents frequency refinement. Defaults to False.

        Returns:
            A tuple containing the estimated respiratory rate and heart rate in Hz.

        Warnings:
            If the segment length is larger than the data length, a warning is issued and the segment length is set to
            the data length. If the overlap is larger than the segment length, a warning is issued and the overlap is
            set to segment length - 1.

        """
        pixel_impedance = eit_data.pixel_impedance.copy().astype(np.float32)
        summed_impedance = np.nansum(pixel_impedance, axis=(1, 2))
        len_segment = int(self.welch_window * eit_data.sample_frequency)

        if len(summed_impedance) < len_segment:
            if not suppress_length_warnings:
                warnings.warn(
                    "The Welch window is longer than the data. Reducing the window length to the lenght of the data.",
                    UserWarning,
                    stacklevel=2,
                )
            len_segment = len(summed_impedance)

        len_overlap = int(len_segment * self.welch_overlap)

        hann_window = signal.windows.hann(len_segment, sym=False)
        frequencies, total_power = signal.welch(
            summed_impedance,
            eit_data.sample_frequency,
            nperseg=len_segment,
            noverlap=len_overlap,
            detrend="constant",
            window=hann_window,
        )

        normalized_total_power = total_power / np.sum(total_power)

        pixel_impedance[:, np.all(pixel_impedance == 0, axis=0)] = np.nan

        _, pixel_power_spectra = signal.welch(
            pixel_impedance,
            eit_data.sample_frequency,
            nperseg=len_segment,
            noverlap=len_overlap,
            detrend="constant",
            axis=0,
            window=hann_window,
        )

        normalized_power_spectra = np.divide(pixel_power_spectra, np.nansum(pixel_power_spectra, axis=0, keepdims=True))
        average_normalized_pixel_power = np.nanmean(normalized_power_spectra, axis=(1, 2))

        diff_total_averaged_power = average_normalized_pixel_power - normalized_total_power

        estimated_respiratory_rate = self._estimate_and_refine_freq(
            frequencies, total_power, self.min_respiratory_rate, self.max_respiratory_rate, suppress_edge_case_warning
        )
        estimated_heart_rate = self._estimate_and_refine_freq(
            frequencies, diff_total_averaged_power, self.min_heart_rate, self.max_heart_rate, suppress_edge_case_warning
        )

        if captures is not None:
            captures["frequencies"] = frequencies
            captures["normalized_total_power"] = normalized_total_power
            captures["average_normalized_pixel_power"] = average_normalized_pixel_power
            captures["diff_total_averaged_power"] = diff_total_averaged_power
            captures["estimated_respiratory_rate"] = estimated_respiratory_rate
            captures["estimated_heart_rate"] = estimated_heart_rate

        return estimated_respiratory_rate, estimated_heart_rate

    def _estimate_and_refine_freq(
        self,
        frequencies: np.ndarray,
        power: np.ndarray,
        min_rate: float,
        max_rate: float,
        suppress_edge_case_warning: bool,
    ) -> float:
        range_indices = np.nonzero((frequencies >= min_rate) & (frequencies <= max_rate))

        frequency_range = frequencies[range_indices]
        power_range = power[range_indices]
        index_max_power = np.argmax(power_range)

        if self.refine_estimated_frequency and (index_max_power == 0 or index_max_power == len(power_range) - 1):
            # If the maximum power is at the edge of the range, we cannot refine the frequency
            if not suppress_edge_case_warning:
                warnings.warn(
                    "Maximum power is at the edge of the range, cannot refine the frequency.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            estimated_rate = frequency_range[index_max_power]

        elif self.refine_estimated_frequency:
            y0, y1, y2 = power_range[index_max_power - 1 : index_max_power + 2]
            x0, x1, x2 = frequency_range[index_max_power - 1 : index_max_power + 2]
            # Parabolic interpolation formula:
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                delta = 0.5 * (y0 - y2) / denom
                estimated_rate = x1 + delta * (x2 - x0) / 2
            else:
                estimated_rate = x1
        else:
            estimated_rate: float = frequency_range[index_max_power]
        return estimated_rate

    @property
    def plotting(self) -> "RateDetectionPlotting":
        """A utility class for plotting the the results of the RateDetection algorithm.

        The `plotting.plot(**captures)` method can be used to plot the results of the algorithm. It takes the captured
        variables from the `apply` method as keyword arguments.

        Example:
        ```python
        >>> rd = RateDetection("adult")
        >>> captures = {}
        >>> estimated_respiratory_rate, estimated_heart_rate = rd.detect_respiratory_heart_rate(eit_data, captures)
        >>> fig = rd.plotting(**captures)
        >>> fig.savefig(...)
        ```

        """
        from eitprocessing.plotting.rate_detection import RateDetectionPlotting

        return RateDetectionPlotting(self)


DEFAULT_RATE_LIMITS = {
    "min_heart_rate": {"adult": 40 / MINUTE, "neonate": 90 / MINUTE},
    "max_heart_rate": {"adult": 200 / MINUTE, "neonate": 210 / MINUTE},
    "min_respiratory_rate": {"adult": 6 / MINUTE, "neonate": 15 / MINUTE},
    "max_respiratory_rate": {"adult": 60 / MINUTE, "neonate": 85 / MINUTE},
}
