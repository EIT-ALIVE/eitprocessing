import itertools
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import signal


duration: TypeAlias = float
index: TypeAlias = int


@dataclass
class Breath:
    start_index: int
    middle_index: int
    end_index: int


@dataclass
class BreathDetection:
    sample_frequency: float
    minimum_distance: duration
    averaging_window_length: duration = 30
    averaging_window_fun: Callable[[int], ArrayLike] | None = None
    amplitude_cutoff_fraction: float | None = 0.25

    def _find_features(
        self, data: NDArray, invert: float = False
    ) -> tuple[NDArray, NDArray]:
        """
        Find features (peaks or valleys) in the data.

        This method finds features (either peaks or valleys) in the data using
        the `scipy.signal.find_peaks()` function. The minimum distance (in
        time) between peaks is determined by the `minimum_distance` attribute.

        To find peaks, `invert` should be False. To find valleys, `invert`
        should be True, which flips the data before finding peaks.

        Args:
            data_ (NDArray): _description_
            invert (float, optional): _description_. Defaults to False.

        Returns:
            tuple[NDArray, NDArray]: _description_
        """

        data_ = -data if invert else data
        naive_feature_indices, _ = signal.find_peaks(
            data_, distance=self.minimum_distance * self.sample_frequency
        )

        naive_feature_values = data[naive_feature_indices]

        return naive_feature_indices, naive_feature_values

    def _remove_outliers(
        self,
        feature_indices: NDArray,
        feature_values: NDArray,
        operator_: Callable,
        moving_average: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Remove outlier peaks/valleys.

        This method removes any feature (peak/valley) if the value of that
        feature is below the moving average of the data at said feature.

        The `operator_` argument should be a callable that takes two arguments,
        and returns a boolean. In most cases, any of the `lt`, `le`, `gt` or
        `ge` functions from the `operator` module suffice.

        Args:
            feature_indices (NDArray): indices of the previously detected peaks
                or valleys.
            feature_values (NDArray): the vallues of the previously detected
                peaks or valleys.
            operator_ (Callable): operator to use when comparing to the moving
                average.
            moving_average (NDArray): _description_

        Returns:
            tuple[NDArray, NDArray]: _description_
        """
        feature_outliers = np.argwhere(
            operator_(feature_values, moving_average[feature_indices])
        )
        feature_indices = np.delete(feature_indices, feature_outliers)
        feature_values = np.delete(feature_values, feature_outliers)
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
            peak_indices (NDArray): indices of the previously detected peaks
            peak_values (NDArray):
            valley_indices (NDArray): indices of the previously detected
                valleys
            valley_values (NDArray):
            data (NDArray): the data in which the peaks/valleys are detected
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
            peak_indices (NDArray): indices of the previously detected peaks
            peak_values (NDArray): the values of the previously detected peaks
            valley_indices (NDArray): indices of the previously detected valleys
            valley_values (NDArray): the values of the previously detected valleys

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

        Returns:
            NDArray: _description_
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

    def find_breaths(self, data: NDArray):
        moving_average = self._calculate_moving_average(data)

        peak_indices, peak_values = self._find_features(data)
        peak_indices, peak_values = self._remove_outliers(
            peak_indices,
            peak_values,
            operator_=operator.lt,
            moving_average=moving_average,
        )

        valley_indices, valley_values = self._find_features(data, invert=True)
        valley_indices, valley_values = self._remove_outliers(
            valley_indices,
            valley_values,
            operator_=operator.gt,
            moving_average=moving_average,
        )

        (
            peak_indices,
            peak_values,
            valley_indices,
            valley_values,
        ) = self._remove_edge_cases(
            peak_indices,
            peak_values,
            valley_indices,
            valley_values,
            data,
            moving_average,
        )

        peak_indices, peak_values, valley_indices, valley_values = self._remove_doubles(
            peak_indices, peak_values, valley_indices, valley_values
        )

        (
            peak_indices,
            peak_values,
            valley_indices,
            valley_values,
        ) = self._remove_low_amplitudes(
            peak_indices, peak_values, valley_indices, valley_values
        )

        return [
            Breath(start, middle, end)
            for middle, (start, end) in zip(
                peak_indices, itertools.pairwise(valley_indices)
            )
        ]
