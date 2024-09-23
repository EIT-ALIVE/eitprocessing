from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from eitprocessing.datahandling.breath import Breath
from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.breath_detection import BreathDetection


@dataclass
class PixelInflation:
    """Algorithm for detecting timing of pixel inflation and deflation in pixel impedance data.

    This algorithm detects the position of start inflation, end inflation/start deflation and
    end deflation in pixel impedance data. It uses BreathDetection to find the global start and end
    of inspiration and expiration. These points are then used to find the start/end of pixel
    inflation/deflation in pixel impedance data.

    Examples:
    >>> pi = PixelInflation()
    >>> eit_data = sequence.eit_data['raw']
    >>> continuous_data = sequence.continuous_data['global_impedance_(raw)']
    >>> pixel_inflations = pi.find_pixel_inflations(eit_data, continuous_data, sequence)

    Args:
    breath_detection_kwargs (dict): A dictionary of keyword arguments for breath detection.
        The available keyword arguments are:
        minimum_duration: minimum expected duration of breaths, defaults to 2/3 of a second
        averaging_window_duration: duration of window used for averaging the data, defaults to 15 seconds
        averaging_window_function: function used to create a window for averaging the data, defaults to np.blackman
        amplitude_cutoff_fraction: fraction of the median amplitude below which breaths are removed, defaults to 0.25
        invalid_data_removal_window_length: window around invalid data in which breaths are removed, defaults to 0.5
        invalid_data_removal_percentile: the nth percentile of values used to remove outliers, defaults to 5
        invalid_data_removal_multiplier: the multiplier used to remove outliers, defaults to 4
    """

    breath_detection_kwargs: dict = field(default_factory=dict)

    def find_pixel_inflations(
        self,
        eit_data: EITData,
        continuous_data: ContinuousData,
        sequence: Sequence | None = None,
        store: bool | None = None,
        result_label: str = "pixel inflations",
    ) -> IntervalData:
        """Find pixel inflations in the data.

        This method finds the pixel start/end of inflation/deflation
        based on the start/end of inspiration/expiration as detected
        in the continuous data.

        If pixel impedance is in phase (within 180 degrees) with the continuous data,
        the start of inflation of that pixel is defined as the local minimum between
        two end-inspiratory points in the continuous signal.
        The end of deflation of that pixel is defined as the local minimum between two
        consecutive end-inspiratory points in the continuous data.
        The end of inflation of that pixel is defined as the local maximum between
        the start of inflation and end of deflation of that pixel.

        If pixel impedance is out of phase with the continuous signal,
        the start of inflation of that pixel is defined as the local maximum between
        two global end-inspiration points.
        The end of deflation of that pixel is defined as the local maximum between two
        consecutive end-inspiratory points in the continuous data.
        The end of inflation of that pixel is defined as the local minimum between
        the start of inflation and end of deflation of that pixel.

        Pixel inflations are constructed as a valley-peak-valley combination,
        representing the start of inflation, the end of inflation/start of
        deflation, and end of deflation.


        Args:
            eit_data: EITData to apply the algorithm to
            continuous_data: ContinuousData to use for global breath detection
            result_label: label of the returned IntervalData object, defaults to `'pixel inflations'`.
            sequence: optional, Sequence that contains the object to detect pixel inflations in,
            and/or to store the result in.
            store: whether to store the result in the sequence, defaults to `True` if a Sequence if provided.

        Returns:
            An IntervalData object containing Breath objects.
        """
        if store is None and isinstance(sequence, Sequence):
            store = True

        if store and sequence is None:
            msg = "Can't store the result if no Sequence is provided."
            raise RuntimeError(msg)

        if store and not isinstance(sequence, Sequence):
            msg = "To store the result a Sequence dataclass must be provided."
            raise ValueError(msg)

        bd_kwargs = self.breath_detection_kwargs.copy()
        breath_detection = BreathDetection(**bd_kwargs)
        breaths = breath_detection.find_breaths(continuous_data)

        indices_breath_middles = np.searchsorted(eit_data.time, [breath.middle_time for breath in breaths.values])

        _, n_rows, n_cols = eit_data.pixel_impedance.shape

        from eitprocessing.parameters.tidal_impedance_variation import TIV

        pixel_tiv_with_continuous_data_timing = TIV().compute_pixel_parameter(
            eit_data,
            continuous_data,
            sequence,
            tiv_method="inspiratory",
            tiv_timing="continuous",
        )

        mean_tiv_pixel = np.nanmean(pixel_tiv_with_continuous_data_timing, axis=0)
        time = eit_data.time
        pixel_impedance = eit_data.pixel_impedance

        pixel_inflations = np.full((len(breaths), n_rows, n_cols), None)

        for row in range(n_rows):
            for col in range(n_cols):
                mean_tiv = mean_tiv_pixel[row, col]

                if mean_tiv == 0.0:
                    continue

                if mean_tiv < 0:
                    start_func, middle_func = np.argmax, np.argmin
                else:
                    start_func, middle_func = np.argmin, np.argmax

                start = self._find_extreme_indices(pixel_impedance, indices_breath_middles, row, col, start_func)
                end = start[1:]
                middle = self._find_extreme_indices(pixel_impedance, start, row, col, middle_func)

                # TODO discuss; this block of code is implemented to prevent noisy pixels from breaking the code.
                # Quick solve is to make entire breath object None if any breath in a pixel does not have
                # consecutive start, middle and end.
                # However, this might cause problems elsewhere.

                if (start[:-1] >= middle).any() or (middle >= end).any():
                    inflations = None
                else:
                    start = start[:-1]
                    inflations = self._construct_inflations(start, middle, end, time)
                pixel_inflations[:, row, col] = inflations

        pixel_inflations_container = IntervalData(
            label=result_label,
            name="Pixel in- and deflation timing as determined by PixelInflation",
            unit=None,
            category="breath",
            intervals=[
                (time[indices_breath_middles[i]], time[indices_breath_middles[i + 1]])
                for i in range(len(indices_breath_middles) - 1)
            ],
            values=pixel_inflations,
            parameters={self.breath_detection_kwargs},
            derived_from=[eit_data],
        )
        if store:
            sequence.interval_data.add(pixel_inflations_container)

        return pixel_inflations_container

    def _construct_inflations(self, start: list, middle: list, end: list, time: np.ndarray) -> list:
        inflations = [Breath(time[s], time[m], time[e]) for s, m, e in zip(start, middle, end, strict=True)]
        # First and last inflation are not detected by definition (need two breaths to find one inflation)
        return [None, *inflations, None]

    def _find_extreme_indices(
        self,
        data: np.ndarray,
        times: np.ndarray,
        row: int,
        col: int,
        function: Callable,
    ) -> np.ndarray:
        return np.array([function(data[times[i] : times[i + 1], row, col]) + times[i] for i in range(len(times) - 1)])
