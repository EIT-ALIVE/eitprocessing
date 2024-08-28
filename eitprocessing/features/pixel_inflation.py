"""Dataclass for pixel inflation detection."""

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
    pi = PixelInflation()
    eit_data = sequence.eit_data['raw']
    continuous_data = sequence.continuous_data['global_impedance_(raw)']
    pixel_inflations = pi.find_pixel_inflations(eit_data, continuous_data, sequence)
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
        in ContinuousData.

        If pixel inflation is in phase with inspiration in the continuous signal,
        the pixel start of inflation is defined as the local minimum between
        two end-inspiration points in the continuous signal.

        Pixel end of deflation is defined as the local minimum between the
        consecutive two end-inspiration points in the continuous signal.

        Pixel end of inflation is defined as the local maximum between
        pixel start of inflation and end of deflation.

        If pixel inflation is out of phase with inspiration in the continuous signal,
        the pixel start of inflation is defined as the local maximum between
        two global end-inspiration points.

        Pixel inflations are constructed as a valley-peak-valley combination,
        representing the start of inflation, the end of inflation/start of
        deflation, and end of deflation.


        Args:
            sequence: the sequence that contains the data
            eit_data: EITData to apply the algorithm to
            continuous_data: ContinuousData to use for global breath detection
            result_label: label of the returned IntervalData object, defaults to `'pixel inflations'`.
            sequence: optional, Sequence that contains the object to detect pixel inflations in,
            and/or to store the result in.
            store: whether to store the result in the sequence, defaults to `True` if a Sequence if provided.

        Returns:
            An IntervalData object containing Breath objects.
        """
        if store is None and sequence:
            store = True

        if store and sequence is None:
            msg = "Can't store the result if no Sequence is provided."
            raise RuntimeError(msg)

        bd_kwargs = self.breath_detection_kwargs.copy()
        breath_detection = BreathDetection(**bd_kwargs)
        breaths = breath_detection.find_breaths(continuous_data)

        middle_times = np.searchsorted(eit_data.time, [breath.middle_time for breath in breaths.values])

        _, rows, cols = eit_data.pixel_impedance.shape

        from eitprocessing.parameters.tidal_impedance_variation import TIV

        tiv_result_pixel_inspiratory_global_timing = TIV().compute_pixel_parameter(
            eit_data,
            continuous_data,
            sequence,
            tiv_method="inspiratory",
            tiv_timing="continuous",
        )

        mean_tiv_pixel = np.nanmean(tiv_result_pixel_inspiratory_global_timing, axis=0)
        time = eit_data.time
        pixel_impedance = eit_data.pixel_impedance

        pixel_inflations = np.empty((len(breaths), rows, cols), dtype=object)

        for row in range(rows):
            for col in range(cols):
                mean_value = mean_tiv_pixel[row, col]
                middle_times_range = middle_times

                if mean_value != 0.0:
                    if mean_value < 0:
                        mode_start, mode_middle = np.argmax, np.argmin
                    else:
                        mode_start, mode_middle = np.argmin, np.argmax

                    start = _find_extreme_indices(pixel_impedance, middle_times_range, row, col, mode_start)
                    end = start[1:]
                    middle = _find_extreme_indices(pixel_impedance, start, row, col, mode_middle)

                    ## To discuss: this block of code is implemented to prevent noisy pixels from breaking the code.
                    # Quick solve is to make entire breath object None if any breath in a pixel does not have
                    # consecutive start, middle and end.
                    # However, this might cause problems elsewhere.

                    if (start[:-1] >= middle).any() or (middle >= end).any():
                        inflations = None
                    else:
                        inflations = _compute_inflations(start, middle, end, time)
                else:
                    inflations = None

                pixel_inflations[:, row, col] = inflations

        pixel_inflations_container = IntervalData(
            label=result_label,
            name="Pixel in- and deflation timing as determined by PixelInflation",
            unit=None,
            category="breath",
            intervals=[(time[middle_times[i]], time[middle_times[i + 1]]) for i in range(len(middle_times) - 1)],
            values=pixel_inflations,
            parameters={},
            derived_from=[eit_data],
        )
        if store:
            sequence.interval_data.add(pixel_inflations_container)

        return pixel_inflations_container


def _compute_inflations(start: list, middle: list, end: list, time: np.ndarray) -> list:
    inflations = [Breath(time[s], time[m], time[e]) for s, m, e in zip(start[:-1], middle, end, strict=True)]
    # First and last inflation are not detected by definition (need two breaths to find one inflation)
    return [None, *inflations, None]


def _find_extreme_indices(data: np.ndarray, times: np.ndarray, row: int, col: int, mode: Callable) -> np.ndarray:
    return np.array([mode(data[times[i] : times[i + 1], row, col]) + times[i] for i in range(len(times) - 1)])
