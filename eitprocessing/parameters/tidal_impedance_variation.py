import itertools
import sys
import warnings
from dataclasses import InitVar, dataclass, field
from functools import singledispatchmethod
from typing import Final, Literal, NoReturn

import numpy as np

from eitprocessing.datahandling.breath import Breath
from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.datahandling.sparsedata import SparseData
from eitprocessing.features.breath_detection import BreathDetection
from eitprocessing.features.pixel_breath import PixelBreath
from eitprocessing.parameters import ParameterCalculation

_SENTINEL_PIXEL_BREATH: Final = PixelBreath()
_SENTINEL_BREATH_DETECTION: Final = BreathDetection()


def _sentinel_pixel_breath() -> PixelBreath:
    # Returns a sential of a PixelBreath, which only exists to signal that the default value for pixel_breath was used.
    return _SENTINEL_PIXEL_BREATH


def _sentinel_breath_detection() -> BreathDetection:
    # Returns a sential of a BreathDetection, which only exists to signal that the default value for breath_detection
    # was used.
    return _SENTINEL_BREATH_DETECTION


@dataclass
class TIV(ParameterCalculation):
    """Compute the tidal impedance variation (TIV) per breath."""

    method: Literal["extremes"] = "extremes"
    breath_detection: BreathDetection = field(default_factory=_sentinel_breath_detection)
    breath_detection_kwargs: InitVar[dict | None] = None

    # The default is a sentinal that will be replaced in __post_init__
    pixel_breath: PixelBreath = field(default_factory=_sentinel_pixel_breath)

    def __post_init__(self, breath_detection_kwargs: dict | None) -> None:
        if self.method != "extremes":
            msg = f"Method {self.method} is not implemented. The method must be 'extremes'."
            raise NotImplementedError(msg)

        if breath_detection_kwargs is not None:
            if self.breath_detection is not _SENTINEL_BREATH_DETECTION:
                msg = (
                    "`breath_detection_kwargs` is deprecated, and can't be used at the same time as `breath_detection`."
                )
                raise TypeError(msg)

            self.breath_detection = BreathDetection(**breath_detection_kwargs)
            warnings.warn(
                "`breath_detection_kwargs` is deprecated and will be removed soon. "
                "Replace with `breath_detection=BreathDetection(**breath_detection_kwargs)`.",
                DeprecationWarning,
            )

        if self.pixel_breath is _SENTINEL_PIXEL_BREATH:
            # If no value was provided at initialization, PixelBreath should use the same BreathDetection object as TIV.
            # However, a default factory cannot be used because it can't access self.breath_detection. The sentinal
            # object is replaced here (only if pixel_breath was not provided) with the correct BreathDetection object.
            self.pixel_breath = PixelBreath(breath_detection=self.breath_detection)

    @singledispatchmethod
    def compute_parameter(
        self,
        data: ContinuousData | EITData,
    ) -> NoReturn:
        """Compute the tidal impedance variation per breath on either ContinuousData or EITData, depending on the input.

        Args:
            data: either continuous_data or eit_data to compute TIV on.
        """
        msg = f"This method is implemented for ContinuousData or EITData, not {type(data)}."
        raise TypeError(msg)

    @compute_parameter.register(ContinuousData)
    def compute_continuous_parameter(
        self,
        continuous_data: ContinuousData,
        tiv_method: Literal["inspiratory", "expiratory", "mean"] = "inspiratory",
        sequence: Sequence | None = None,
        store: bool | None = None,
        result_label: str = "continuous_tivs",
    ) -> SparseData:
        """Compute the tidal impedance variation per breath.

        Args:
            continuous_data: The ContinuousData to compute the TIV on.
            tiv_method: The label of which part of the breath the TIV
                should be determined on (inspiratory, expiratory, or mean).
                Defaults to 'inspiratory'.
            sequence: optional, Sequence that contains the object to detect TIV on,
            and/or to store the result in.
            store: whether to store the result in the sequence, defaults to `True` if a Sequence if provided.
            result_label: label of the returned SparseData object, defaults to `'continuous_tivs'`.

        Returns:
            A SparseData object with the computed TIV values.

        Raises:
            RuntimeError: If store is set to true but no sequence is provided.
            ValueError: If the provided sequence is not an instance of the Sequence dataclass.
            ValueError: If tiv_method is not one of 'inspiratory', 'expiratory', or 'mean'.
        """
        if store is None and isinstance(sequence, Sequence):
            store = True

        if store and sequence is None:
            msg = "Can't store the result if no Sequence is provided."
            raise RuntimeError(msg)

        if store and not isinstance(sequence, Sequence):
            msg = "To store the result a Sequence dataclass must be provided."
            raise ValueError(msg)

        if tiv_method not in {"inspiratory", "expiratory", "mean"}:
            msg = f"Invalid tiv_method: {tiv_method}. Must be one of 'inspiratory', 'expiratory', or 'mean'."
            raise ValueError(msg)

        breaths = self._detect_breaths(continuous_data)

        tiv_values = self._calculate_tiv_values(
            continuous_data.values,
            continuous_data.time,
            breaths.values,
            tiv_method,
            tiv_timing="continuous",
        )
        tiv_container = SparseData(
            label=result_label,
            name="Continuous tidal impedance variation",
            unit=None,
            category="impedance difference",
            time=[breath.middle_time for breath in breaths.values if breath is not None],
            description="Tidal impedance variation determined on continuous data",
            derived_from=[continuous_data],
            values=tiv_values,
        )
        if store:
            sequence.sparse_data.add(tiv_container)

        return tiv_container

    @compute_parameter.register(EITData)
    def compute_pixel_parameter(
        self,
        eit_data: EITData,
        continuous_data: ContinuousData,
        sequence: Sequence,
        tiv_method: Literal["inspiratory", "expiratory", "mean"] = "inspiratory",
        tiv_timing: Literal["pixel", "continuous"] = "pixel",
        store: bool | None = None,
        result_label: str = "pixel_tivs",
    ) -> SparseData:
        """Compute the tidal impedance variation per breath on pixel level.

        Args:
            sequence: The sequence containing the data.
            eit_data: The eit pixel level data to determine the TIV of.
            continuous_data: The continuous data to determine the continuous data breaths or pixel level breaths.
            tiv_method: The label of which part of the breath the TIV should be determined on
                        (inspiratory, expiratory or mean). Defaults to 'inspiratory'.
            tiv_timing: The label of which timing should be used to compute the TIV, either based on breaths
                        detected in continuous data ('continuous') or based on pixel breaths ('pixel').
                        Defaults to 'pixel'.
            result_label: label of the returned IntervalData object, defaults to `'pixel_tivs'`.
            store: whether to store the result in the sequence, defaults to `True` if a Sequence if provided.

        Returns:
            A SparseData object with the computed TIV values.

        Raises:
            RuntimeError: If store is set to true but no sequence is provided.
            ValueError: If the provided sequence is not an instance of the Sequence dataclass.
            ValueError: If tiv_method is not one of 'inspiratory', 'expiratory', or 'mean'.
            ValueError: If tiv_timing is not one of 'continuous' or 'pixel'.
        """
        if store is None and isinstance(sequence, Sequence):
            store = True

        if store and sequence is None:
            msg = "Can't store the result if no Sequence is provided."
            raise RuntimeError(msg)

        if store and not isinstance(sequence, Sequence):
            msg = "To store the result a Sequence dataclass must be provided."
            raise ValueError(msg)

        if tiv_method not in ["inspiratory", "expiratory", "mean"]:
            msg = f"Invalid {tiv_method}. The tiv_method must be either 'inspiratory', 'expiratory' or 'mean'."
            raise ValueError(msg)

        if tiv_timing not in ["continuous", "pixel"]:
            msg = f"Invalid {tiv_timing}. The tiv_timing must be either 'continuous' or 'pixel'."
            raise ValueError(msg)

        data = eit_data.pixel_impedance
        _, n_rows, n_cols = data.shape

        if tiv_timing == "pixel":
            pixel_breaths = self._detect_pixel_breaths(
                eit_data,
                continuous_data,
                sequence,
                store=False,
            )  # Set store to false as to not save these pixel breaths as IntervalData.
            # Check if pixel_breaths.values is empty
            breath_data = (
                np.empty((0, n_rows, n_cols)) if not len(pixel_breaths.values) else np.stack(pixel_breaths.values)
            )
            ## TODO: replace with breath_data = pixel_breaths.values when IntervalData works with 3D array
        else:  # tiv_timing == "continuous"
            global_breaths = self._detect_breaths(
                continuous_data,
            )
            breath_data = global_breaths.values

        number_of_breaths = len(breath_data)
        all_pixels_tiv_values = np.full((number_of_breaths, n_rows, n_cols), None, dtype=object)
        all_pixels_breath_timings = np.full((number_of_breaths, n_rows, n_cols), None, dtype=object)

        for row, col in itertools.product(range(n_rows), range(n_cols)):
            time_series = data[:, row, col]
            breaths = breath_data[:, row, col] if tiv_timing == "pixel" else breath_data
            pixel_tiv_values = self._calculate_tiv_values(
                time_series,
                eit_data.time,
                breaths,
                tiv_method,
                tiv_timing,
            )
            # Get the middle times of each breath where breath is not None
            pixel_breath_timings = [breath.middle_time for breath in breaths if breath is not None]

            # Store these in all_pixels_breath_timings, ensuring they match the expected shape
            all_pixels_breath_timings[: len(pixel_breath_timings), row, col] = pixel_breath_timings

            all_pixels_tiv_values[:, row, col] = pixel_tiv_values

        tiv_container = SparseData(
            label=result_label,
            name="Pixel tidal impedance variation",
            unit=None,
            category="impedance difference",
            time=list(all_pixels_breath_timings),
            description="Tidal impedance variation determined on pixel impedance",
            derived_from=[eit_data],
            values=list(all_pixels_tiv_values.astype(float)),
        )

        if store:
            sequence.sparse_data.add(tiv_container)

        return tiv_container

    def _detect_breaths(self, data: ContinuousData) -> IntervalData:
        return self.breath_detection.find_breaths(data)

    def _detect_pixel_breaths(
        self,
        eit_data: EITData,
        continuous_data: ContinuousData,
        sequence: Sequence,
        store: bool,
    ) -> IntervalData:
        return self.pixel_breath.find_pixel_breaths(
            eit_data,
            continuous_data,
            result_label="pixel breaths",
            sequence=sequence,
            store=store,
        )

    def _calculate_tiv_values(
        self,
        data: np.ndarray,
        time: np.ndarray,
        breaths: list[Breath],
        tiv_method: str,
        tiv_timing: str,
    ) -> list:
        # Filter out None breaths
        valid_breaths = [breath for breath in breaths if breath is not None]

        # If there are no valid breaths, return a list of None with the same length as the number of breaths
        if not valid_breaths:
            return [None] * len(breaths)

        start_indices = np.searchsorted(time, [breath.start_time for breath in breaths if breath is not None])
        middle_indices = np.searchsorted(time, [breath.middle_time for breath in breaths if breath is not None])
        end_indices = np.searchsorted(time, [breath.end_time for breath in breaths if breath is not None])

        if tiv_method == "inspiratory":
            tiv_values = np.squeeze(np.diff(data[[start_indices, middle_indices]], axis=0), axis=0)

        elif tiv_method == "expiratory":
            tiv_values = np.squeeze(np.diff(data[[end_indices, middle_indices]], axis=0), axis=0)

        elif tiv_method == "mean":
            mean_outer_values = data[[start_indices, end_indices]].mean(axis=0)
            end_inspiratory_values = data[middle_indices]
            tiv_values = end_inspiratory_values - mean_outer_values
        else:
            msg = f"`tiv_method` ({tiv_method}) not valid."
            exc = ValueError(msg)
            if sys.version_info >= (3, 11):
                exc.add_note("Valid value for `tiv_method` are 'inspiratory', 'expiratory' and 'mean'.")
            raise exc

        if tiv_timing == "pixel":
            tiv_values = [None, *tiv_values, None]

        return tiv_values
