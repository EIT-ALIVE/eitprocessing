import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final
from typing import Literal
from typing import NamedTuple
from typing import TypeAlias
import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import signal
from eitprocessing.filters.butterworth_filters import HighPassFilter


duration: TypeAlias = float
index: TypeAlias = int
MINUTE: Final = 60


class Breath(NamedTuple):
    """Indicates the start, middle and end of a single breath."""

    start_index: index
    middle_index: index
    end_index: index


@dataclass
class BreathDetection:
    """Algorithm for detecting breaths in data representing respiration.

    This algorithm detects the position of breaths in data by detecting valleys
    (local minimum values) and peaks (local maximum values) in data. When
    initializing BreathDetection, the sample frequency of the data and the
    minimum duration of a breath have to be provided. The minimum duration
    should be short enough to include the shortest expected breath in the data.

    Args:
        -

    Examples:
    >>> bd = BreathDetection(sample_frequency=50, minimum_duration=0.5)
    >>> breaths = bd.find_breaths(global_impedance)
    """

    sample_frequency: float
    minimum_distance: duration = 2 / 3
    averaging_window_length: duration = 15
    averaging_window_fun: Callable[[int], ArrayLike] | None = np.bartlett
    amplitude_cutoff_fraction: float | None = 0.25
    invalid_data_removal_window_length: duration = 1
    invalid_data_removal_percentile: int = 5
    invalid_data_removal_multiplier: int = 4
    high_pass_filter_cutoff: float | None = 6 / MINUTE

    def _find_features(
        self, data: NDArray, moving_average: NDArray, invert: float = False
    ) -> tuple[NDArray, NDArray]:
        """
        Find features (peaks or valleys) in the data.

        This method finds features (either peaks or valleys) in the data using
        the `scipy.signal.find_peaks()` function. The minimum distance (in
        time) between peaks is determined by the `minimum_distance` attribute.

        To find peaks, `invert` should be False. To find valleys, `invert`
        should be True, which flips the data before finding peaks.

        Args:
            data (NDArray): a 1D array containing the data.
            moving_average (NDArrag): a 1D array containing the moving average
                of the data.
            invert (float, optional): whether to invert the data before
            detecting peaks. Defaults to False.

        Returns:
            A tuple containing two 1D arrays of length N with the indices (int)
            and values (float) of the features, where N is the number of
            features found.
        """

        data_ = -data if invert else data
        moving_average_ = -moving_average if invert else moving_average
        feature_indices, _ = signal.find_peaks(
            data_,
            distance=self.minimum_distance * self.sample_frequency,
            height=moving_average_,
        )

        feature_values = data[feature_indices]

        return feature_indices, feature_values

    def _remove_edge_cases(
        self,
        peak_indices: NDArray,
        peak_values: NDArray,
        valley_indices: NDArray,
        valley_values: NDArray,
        data: NDArray,
        moving_average: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Remove overdetected peaks/valleys at the start and end of the data.

        This method removed a valley at the start of the data, if the data
        before said valley stays below the moving average of the data at said
        valley. Likewise, it removes the last valley if the data after the last
        valley stays below the moving average of the data at said valley. This
        ensures a valley is a true valley, and not just a local minimum while
        the true valley is cut off.

        Then, all peaks that occur before the first and after the last valley
        are removed. This ensures peaks only fall between valleys.

        Args:
            peak_indices (NDArray): indices of the peaks
            peak_values (NDArray): values of the peaks
            valley_indices (NDArray): indices of the valleys
            valley_values (NDArray): values of the valleys
            data (NDArray): the data in which the peaks/valleys were detected
            moving_average (NDArray): the moving average of data

        Returns:
            A tuple (peak_indices, peak_values, valley_indices, valley_values)
            with edge cases removed.
        """
        if max(data[: valley_indices[0]]) < moving_average[valley_indices[0]]:
            # remove the first valley, if the data before that valley is not
            # high enough to be sure it's a valley
            valley_indices = np.delete(valley_indices, 0)
            valley_values = np.delete(valley_values, 0)

        if max(data[valley_indices[-1] :]) < moving_average[valley_indices[-1]]:
            # remove the last valley, if the data after that valley is not high
            # enough to be sure it's a valley
            valley_indices = np.delete(valley_indices, -1)
            valley_values = np.delete(valley_values, -1)

        # remove peaks that come before the first valley
        keep_peaks = peak_indices > valley_indices[0]
        peak_indices = peak_indices[keep_peaks]
        peak_values = peak_values[keep_peaks]

        # remove peak that come after the last valley
        keep_peaks = peak_indices < valley_indices[-1]
        peak_indices = peak_indices[keep_peaks]
        peak_values = peak_values[keep_peaks]

        return peak_indices, peak_values, valley_indices, valley_values

    def _remove_doubles(
        self,
        peak_indices: NDArray,
        peak_values: NDArray,
        valley_indices: NDArray,
        valley_values: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Remove double peaks/valleys.

        This method ensures there is only one peak between valleys, and only
        one valley between peaks. If there are multiple peaks between two
        valleys, the peak with the highest value is kept and the others are
        removed. If there are no peaks between several valleys (i.e. multiple
        valleys between peaks) the valley with the lowest value is kept, while
        the others are removed.

        This method does not remove peaks before the first or after the last
        valley.

        Args:
            peak_indices (NDArray): indices of the peaks
            peak_values (NDArray): values of the peaks
            valley_indices (NDArray): indices of the valleys
            valley_values (NDArray): values of the valleys

        Returns:
            A tuple (peak_indices, peak_values, valley_indices, valley_values)
            with double peaks/valleys removed.
        """

        current_valley_index = 0
        while current_valley_index < len(valley_indices) - 1:
            start_index = valley_indices[current_valley_index]
            end_index = valley_indices[current_valley_index + 1]
            peaks_between_valleys = np.argwhere(
                (peak_indices > start_index) & (peak_indices < end_index)
            )
            if not len(peaks_between_valleys):
                # no peak between valleys, remove highest valley
                delete_valley_index = (
                    current_valley_index
                    if valley_values[current_valley_index]
                    > valley_values[current_valley_index + 1]
                    else current_valley_index + 1
                )
                valley_indices = np.delete(valley_indices, delete_valley_index)
                valley_values = np.delete(valley_values, delete_valley_index)
                continue

            if len(peaks_between_valleys) > 1:
                # multiple peaks between valleys, remove lowest peak
                delete_peak_index = (
                    peaks_between_valleys[0]
                    if peak_values[peaks_between_valleys[0]]
                    < peak_values[peaks_between_valleys[1]]
                    else peaks_between_valleys[1]
                )
                peak_indices = np.delete(peak_indices, delete_peak_index)
                peak_values = np.delete(peak_values, delete_peak_index)
                continue

            current_valley_index += 1

        return peak_indices, peak_values, valley_indices, valley_values

    def _calculate_moving_average(self, data: NDArray) -> NDArray:
        """
        Calculate the moving average of the data.

        The moving average is calculated using a convolution with a window. The
        window length (in seconds) is determined by the attribute
        `averaging_window_duration`. The shape of the window is determined by
        `averaging_window_fun`, which should be a callable that takes an
        integer `M` and returns an array-like sequence containing a window with
        length `M` and area 1.

        Before convolution the data is padded at the start and end with the
        first/last value, to more accurately determine the average at the
        boundaries of the data.

        Args:
            data (NDArray): input data as 1D array

        Returns:
            Moving average as a 1D array with the same length as `data`.
        """

        window_size = (
            int(
                (np.round(self.sample_frequency * self.averaging_window_length / 2)) * 2
            )
            + 1
        )
        if window_size > len(data):
            window_size = int((len(data) - 1) / 2) + 1

        if self.averaging_window_fun:
            window = np.array(self.averaging_window_fun(window_size))
            window = window / np.sum(window)
        else:
            window = np.ones(window_size) / window_size

        padding_length = (window_size - 1) // 2
        padded_data = np.pad(data, padding_length, mode="edge")
        moving_average = np.convolve(padded_data, window, mode="valid")

        return moving_average

    def _remove_low_amplitudes(
        self,
        peak_indices: NDArray,
        peak_values: NDArray,
        valley_indices: NDArray,
        valley_values: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Remove peaks if the amplitude is too low compared to the median
        amplitude.

        The amplitude of a peak is determined as the average vertical distance
        between the peak value and the two valley values besides it. The cutoff
        value for the amplitude is calculated as the median amplitude times
        `amplitude_cutoff_fraction`. Peaks that have an amplitude below the
        cutoff are removed. Then, `_remove_doubles()` is called to remove
        either of the valleys next to the peak.

        If `amplitude_cutoff_fraction` is None, the input is returned
        unchanged.

        Args:
            peak_indices (NDArray): the indices of the peaks
            peak_values (NDArray): the values of the peaks
            valley_indices (NDArray): the indices of the valleys
            valley_vallues (NDArray): the values of the valleys

        Returns:
            A tuple (peak_indices, peak_values, valley_indices, valley_values)
            with low-amplitude breaths removed.
        """
        if not self.amplitude_cutoff_fraction:
            return peak_indices, peak_values, valley_indices, valley_values

        inspiratory_amplitude = peak_values - valley_values[:-1]
        expiratory_amplitude = peak_values - valley_values[1:]
        amplitude = (inspiratory_amplitude + expiratory_amplitude) / 2

        amplitude_cutoff_value = self.amplitude_cutoff_fraction * np.median(amplitude)
        delete_peaks = np.argwhere(amplitude < amplitude_cutoff_value)

        peak_indices = np.delete(peak_indices, delete_peaks)
        peak_values = np.delete(peak_values, delete_peaks)

        peak_indices, peak_values, valley_indices, valley_values = self._remove_doubles(
            peak_indices, peak_values, valley_indices, valley_values
        )

        return peak_indices, peak_values, valley_indices, valley_values

    def _remove_breaths_around_invalid_data(self, breaths: list[Breath], data: NDArray):
        mean = np.mean(data)
        cutoff_low = (
            mean
            - (mean - np.percentile(data, self.invalid_data_removal_percentile))
            * self.invalid_data_removal_multiplier
        )
        cutoff_high = (
            mean
            + (mean + np.percentile(data, 100 - self.invalid_data_removal_percentile))
            * self.invalid_data_removal_multiplier
        )
        outliers = (data < cutoff_low) | (data > cutoff_high)

        window_length = math.ceil(
            self.invalid_data_removal_window_length * self.sample_frequency
        )
        window = np.ones(window_length)

        extended_data_is_zero = np.convolve(outliers, window, mode="same")
        extended_data_is_zero = extended_data_is_zero.astype(bool).astype(int)

        for breath in breaths[:]:
            if np.max(extended_data_is_zero[breath.start_index : breath.end_index]):
                breaths.remove(breath)

        return breaths

    def _get_extreme_values(
        self, peak_indices: NDArray, valley_indices: NDArray, data: NDArray
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        def get_new_features(
            limit_indices: NDArray, target_indices: NDArray, function: Callable
        ):
            for lower_limit, upper_limit in itertools.pairwise(limit_indices):
                features_within_limits = target_indices[
                    (target_indices > lower_limit) & (target_indices < upper_limit)
                ]
                if len(features_within_limits):
                    yield function(data[lower_limit:upper_limit]) + lower_limit

        new_peak_indices = np.array(
            list(get_new_features(valley_indices, peak_indices, np.argmax))
        )
        new_valley_indices = np.array(
            list(get_new_features(peak_indices, valley_indices, np.argmin))
        )
        new_peak_values = data[new_peak_indices]
        new_valley_values = data[new_valley_indices]

        return (
            new_peak_indices,
            new_peak_values,
            new_valley_indices,
            new_valley_values,
        )

    def find_breaths(self, data: NDArray):
        """Find breaths in the data.

        This method attempts to find peaks and valleys in the data in a
        multi-step process. First, it naively finds any peaks that are a
        certain distance apart and higher than the moving average, and
        similarly valleys that are a certain distance apart and below the
        moving average.

        Next, valleys at the start and end of the signal are removed
        to ensure the first and last valleys are actual valleys, and not just
        the start or end of the signal. Peaks before the first or after the
        last valley are removed, to ensure peaks always fall between two
        valleys.

        At this point, it is possible multiple peaks exist between two valleys.
        Lower peaks are removed leaving only the highest peak between two
        valleys. Similarly, multiple valleys between two peaks are reduced to
        only the lowest valley.

        As a last step, breaths with a low amplitude (the average between the
        inspiratory and expiratory amplitudes) are removed.

        Breaths are constructed as a valley-peak-valley combination,
        representing the start of inspiration, the end of inspiration/start of
        expiration, and end of expiration.

        Args:
            data (NDArray): a 1D array containing the data to find breaths in.

        Returns:
            A list of Breath objects.

        """
        if self.high_pass_filter_cutoff:
            filter_ = HighPassFilter(
                cutoff_frequency=self.high_pass_filter_cutoff,
                order=5,
                sample_frequency=self.sample_frequency,
            )
            filtered_data = filter_.apply_filter(data)

            moving_average = self._calculate_moving_average(filtered_data)
            peak_indices, peak_values = self._find_features(
                filtered_data, moving_average
            )
            valley_indices, valley_values = self._find_features(
                filtered_data, moving_average, invert=True
            )

            (
                peak_indices,
                peak_values,
                valley_indices,
                valley_values,
            ) = self._get_extreme_values(peak_indices, valley_indices, data)

        else:
            moving_average = self._calculate_moving_average(data)

            peak_indices, peak_values = self._find_features(data, moving_average)
            valley_indices, valley_values = self._find_features(
                data, moving_average, invert=True
            )

        indices_and_values = (peak_indices, peak_values, valley_indices, valley_values)
        indices_and_values = self._remove_edge_cases(
            *indices_and_values, data, moving_average
        )
        indices_and_values = self._remove_doubles(*indices_and_values)
        indices_and_values = self._remove_low_amplitudes(*indices_and_values)

        peak_indices, _, valley_indices, _ = indices_and_values

        breaths = [
            Breath(start, middle, end)
            for middle, (start, end) in zip(
                peak_indices, itertools.pairwise(valley_indices)
            )
        ]

        breaths = self._remove_breaths_around_invalid_data(breaths, data)

        return breaths
