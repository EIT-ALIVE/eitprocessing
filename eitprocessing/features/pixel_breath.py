import itertools
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
class PixelBreath:
    """Algorithm for detecting timing of pixel breaths in pixel impedance data.

    This algorithm detects the position of start of inspiration, end of inspiration and
    end of expiration in pixel impedance data. It uses BreathDetection to find the global start and end
    of inspiration and expiration. These points are then used to find the start/end of pixel
    inspiration/expiration in pixel impedance data.

    Example:
    ```
    >>> pi = PixelBreath()
    >>> eit_data = sequence.eit_data['raw']
    >>> continuous_data = sequence.continuous_data['global_impedance_(raw)']
    >>> pixel_breaths = pi.find_pixel_breaths(eit_data, continuous_data, sequence)
    ```

    Args:
        breath_detection (BreathDetection): BreathDetection object to use for detecing breaths.
        allow_negative_amplitude (bool): whether to asume out-of-phase pixels have negative amplitude instead.
    """

    breath_detection: BreathDetection = field(default_factory=BreathDetection)
    allow_negative_amplitude: bool = True

    def find_pixel_breaths(
        self,
        eit_data: EITData,
        continuous_data: ContinuousData,
        sequence: Sequence | None = None,
        store: bool | None = None,
        result_label: str = "pixel_breaths",
    ) -> IntervalData:
        """Find pixel breaths in the data.

        This method finds the pixel start/end of inspiration/expiration
        based on the start/end of inspiration/expiration as detected
        in the continuous data.

        If pixel impedance is in phase (within 180 degrees) with the continuous data,
        the start of breath of that pixel is defined as the local minimum between
        two end-inspiratory points in the continuous signal.
        The end of expiration of that pixel is defined as the local minimum between two
        consecutive end-inspiratory points in the continuous data.
        The end of inspiration of that pixel is defined as the local maximum between
        the start of inspiration and end of expiration of that pixel.

        If pixel impedance is out of phase with the continuous signal,
        the start of inspiration of that pixel is defined as the local maximum between
        two end-inspiration points in the continuous data.
        The end of expiration of that pixel is defined as the local maximum between two
        consecutive end-inspiratory points in the continuous data.
        The end of inspiration of that pixel is defined as the local minimum between
        the start of inspiration and end of expiration of that pixel.

        Pixel breaths are constructed as a valley-peak-valley combination,
        representing the start of inspiration, the end of inspiration/start of
        expiration, and end of expiration.

        Args:
            eit_data: EITData to apply the algorithm to
            continuous_data: ContinuousData to use for global breath detection
            result_label: label of the returned IntervalData object, defaults to `'pixel_breaths'`.
            sequence: optional, Sequence that contains the object to detect pixel breaths in,
            and/or to store the result in.
            store: whether to store the result in the sequence, defaults to `True` if a Sequence if provided.

        Returns:
            An IntervalData object containing Breath objects.

        Raises:
            RuntimeError: If store is set to true but no sequence is provided.
            ValueError: If the provided sequence is not an instance of the Sequence dataclass.
        """
        if store is None and isinstance(sequence, Sequence):
            store = True

        if store and sequence is None:
            msg = "Can't store the result if no Sequence is provided."
            raise RuntimeError(msg)

        if store and not isinstance(sequence, Sequence):
            msg = "To store the result a Sequence dataclass must be provided."
            raise ValueError(msg)

        continuous_breaths = self.breath_detection.find_breaths(continuous_data)

        indices_breath_middles = np.searchsorted(
            eit_data.time,
            [breath.middle_time for breath in continuous_breaths.values],
        )

        _, n_rows, n_cols = eit_data.pixel_impedance.shape

        from eitprocessing.parameters.tidal_impedance_variation import TIV

        pixel_tivs = TIV().compute_pixel_parameter(
            eit_data,
            continuous_data,
            sequence,
            tiv_method="inspiratory",
            tiv_timing="continuous",
            store=False,
        )  # Set store to false as to not save these pixel tivs as SparseData.

        pixel_tiv_with_continuous_data_timing = (
            np.empty((0, n_rows, n_cols)) if not len(pixel_tivs.values) else np.stack(pixel_tivs.values)
        )

        # Create a mask to detect slices that are entirely NaN
        all_nan_mask = np.isnan(pixel_tiv_with_continuous_data_timing).all(axis=0)

        # Initialize the mean_tiv_pixel array with NaNs
        mean_tiv_pixel = np.full((n_rows, n_cols), np.nan)

        # Only compute nanmean for slices that are not entirely NaN
        if not all_nan_mask.all():  # Check if there are any valid (non-all-NaN) slices
            mean_tiv_pixel[~all_nan_mask] = np.nanmean(pixel_tiv_with_continuous_data_timing[:, ~all_nan_mask], axis=0)

        time = eit_data.time
        pixel_impedance = eit_data.pixel_impedance

        pixel_breaths = np.full((len(continuous_breaths), n_rows, n_cols), None)

        for row, col in itertools.product(range(n_rows), range(n_cols)):
            mean_tiv = mean_tiv_pixel[row, col]

            if np.std(pixel_impedance[:, row, col]) == 0:
                # pixel has no amplitude
                continue

            if self.allow_negative_amplitude and mean_tiv < 0:
                start_func, middle_func = np.argmax, np.argmin
            else:
                start_func, middle_func = np.argmin, np.argmax

            outsides = self._find_extreme_indices(pixel_impedance, indices_breath_middles, row, col, start_func)
            starts = outsides[:-1]
            ends = outsides[1:]
            middles = self._find_extreme_indices(pixel_impedance, outsides, row, col, middle_func)
            # TODO discuss; this block of code is implemented to prevent noisy pixels from breaking the code.
            # Quick solve is to make entire breath object None if any breath in a pixel does not have
            # consecutive start, middle and end.
            # However, this might cause problems elsewhere.
            if (starts >= middles).any() or (middles >= ends).any():
                pixel_breath = None
            else:
                pixel_breath = self._construct_breaths(starts, middles, ends, time)
            pixel_breaths[:, row, col] = pixel_breath

        intervals = [(breath.start_time, breath.end_time) for breath in continuous_breaths.values]

        pixel_breaths_container = IntervalData(
            label=result_label,
            name="Pixel in- and deflation timing as determined by Pixelbreath",
            unit=None,
            category="breath",
            intervals=intervals,
            values=list(
                pixel_breaths,
            ),  ## TODO: change back to pixel_breaths array when IntervalData works with 3D array
            derived_from=[eit_data],
        )
        if store:
            sequence.interval_data.add(pixel_breaths_container)

        return pixel_breaths_container

    def _construct_breaths(self, start: list[int], middle: list[int], end: list[int], time: np.ndarray) -> list:
        breaths = [Breath(time[s], time[m], time[e]) for s, m, e in zip(start, middle, end, strict=True)]
        # First and last breath are not detected by definition (need two breaths to find one breath)
        return [None, *breaths, None]

    def _find_extreme_indices(
        self,
        pixel_impedance: np.ndarray,
        times: np.ndarray,
        row: int,
        col: int,
        function: Callable,
    ) -> np.ndarray:
        """Finds extreme indices in pixel impedance.

        This method divides the pixel impedance for a single pixel (selected using row and col) into smaller segments
        based on the `times` array. The times array consists of indices to divide these segments. The function iterates
        over each index in the times array to select consecutive segments of pixel impedance.
        For each segment, the method applies the `function` (either np.argmax or np.argmin) to extract an extreme value
        (local maximum or minimum).

        Args:
            pixel_impedance (np.ndarray): The pixel impedance array from which the function will extract values.
                Assumed to be 3-dimensional (e.g., time, rows, and columns).
            times (np.ndarray): 1D array of time indices. These times define the start
                and end of each segment in the pixel impedance.
            row (int): The row index in the pixel impedance
            col (int): The column index in the pixel impedance
            function (Callable): A function that is applied to each segment of data to find
                an extreme value (np.argmax or np.argmin)

        Returns:
            np.ndarray: An array of indices where the extreme values (based on the function)
            are located for each time segment.
        """
        return np.array(
            [function(pixel_impedance[times[i] : times[i + 1], row, col]) + times[i] for i in range(len(times) - 1)],
        )
