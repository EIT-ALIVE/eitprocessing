from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Literal

import numpy as np

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.breath_detection import BreathDetection
from eitprocessing.features.pixel_inflation import PixelInflation
from eitprocessing.parameters import ParameterCalculation


@dataclass
class TIV(ParameterCalculation):
    """Compute the tidal impedance variation (TIV) per breath."""

    method: str = "extremes"
    breath_detection_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        pass

    def _detect_breaths(self, data, sample_frequency, breath_detection_kwargs):
        bd_kwargs = breath_detection_kwargs.copy()
        bd_kwargs["sample_frequency"] = sample_frequency
        breath_detection = BreathDetection(**bd_kwargs)
        return breath_detection.find_breaths(data)

    def _detect_pixel_inflations(self, eit_data, continuous_data, sequence, breath_detection_kwargs):
        bd_kwargs = breath_detection_kwargs.copy()
        bd_kwargs["sample_frequency"] = sample_frequency
        pi = PixelInflation(**bd_kwargs)
        return pi.find_pixel_inflations(eit_data, continuous_data, result_label="pixel inflations", sequence=sequence)

    def _calculate_tiv_values(self, data, time, breaths, tiv_method):
        start_indices = [np.argmax(time == start_time) for start_time in [breath.start_time for breath in breaths]]
        middle_indices = [np.argmax(time == middle_time) for middle_time in [breath.middle_time for breath in breaths]]
        end_indices = [np.argmax(time == end_time) for end_time in [breath.end_time for breath in breaths]]

        if tiv_method == "inspiratory":
            end_inspiratory_values = data[middle_indices]
            start_inspiratory_values = data[start_indices]
            tiv_values = end_inspiratory_values - start_inspiratory_values

        elif tiv_method == "expiratory":
            start_expiratory_values = data[middle_indices]
            end_expiratory_values = data[end_indices]
            tiv_values = start_expiratory_values - end_expiratory_values

        elif tiv_method == "mean":
            start_inspiratory_values = data[start_indices]
            end_inspiratory_values = data[middle_indices]
            end_expiratory_values = data[end_indices]
            tiv_values = end_inspiratory_values - [
                np.mean(k) for k in zip(start_inspiratory_values, end_expiratory_values)
            ]

        return tiv_values

    @singledispatchmethod
    def compute_parameter(
        self,
        data: ContinuousData | EITData,
    ):
        msg = f"This method is implemented for ContinuousData or EITData, not {type(data)}."
        raise TypeError(msg)

    @compute_parameter.register(ContinuousData)
    def compute_global_parameter(
        self,
        continuous_data: ContinuousData,
        tiv_method: Literal["inspiratory", "expiratory", "mean"] = "inspiratory",
    ) -> dict | list[dict]:
        """Compute the tidal impedance variation per breath.

        Args:
            sequence: the sequence containing the data.
            data_label: the label of the continuous data in the sequence to determine the TIV of.
            tiv_method: The label of which part of the breath the TIV
            should be determined on (inspiratory, expiratory or mean).
            Defaults to 'inspiratory'
        """
        if self.method != "extremes":
            msg = f"Method {self.method} is not implemented."
            raise NotImplementedError(msg)

        eitdata = next(filter(lambda x: isinstance(x, EITData), continuous_data.derived_from))

        breaths = self._detect_breaths(continuous_data, eitdata.sample_frequency, self.breath_detection_kwargs)
        return self._calculate_tiv_values(
            continuous_data.values,
            continuous_data.time,
            breaths.values,
            tiv_method,
        )

    @compute_parameter.register(EITData)
    def compute_pixel_parameter(
        self,
        eit_data: EITData,
        continuous_data: ContinuousData,
        sequence: Sequence,
        tiv_method: Literal["inspiratory", "expiratory", "mean"] = "inspiratory",
        tiv_timing: Literal["pixel", "continuous"] = "pixel",
    ) -> dict | list[dict]:
        """Compute the tidal impedance variation per breath on pixel level.

        Args:
            sequence: The sequence containing the data.
            eit_data: The eit pixel level date to determine the TIV of.
            continuous_data: The continuous data to determine the continuous data breaths or pixel level inflations.
            tiv_method: The label of which part of the breath the TIV should be determined on
                        (inspiratory, expiratory or mean). Defaults to 'inspiratory'.
            tiv_timing: The label of which timing should be used to compute the TIV, either based on breaths
                        detected in continuous data ('continuous') or based on pixel inflations ('pixel').
                        Defaults to 'pixel'.

        Raises:
            NotImplementedError: If the method is not 'extremes'.
            ValueError: If tiv_method is not one of 'inspiratory', 'expiratory', or 'mean'.
            ValueError: If tiv_timing is not one of 'continuous' or 'pixel'.
        """
        if self.method != "extremes":
            msg = f"Method {self.method} is not implemented."
            raise NotImplementedError(msg)

        if tiv_method not in ["inspiratory", "expiratory", "mean"]:
            msg = "tiv_method must be either 'inspiratory', 'expiratory' or 'mean'"
            raise ValueError(msg)

        if tiv_timing not in ["continuous", "pixel"]:
            msg = "tiv_timing must be either 'continuous' or 'pixel'"
            raise ValueError(msg)

        data = eit_data.pixel_impedance
        _, rows, cols = data.shape

        if tiv_timing == "pixel":
            pixel_inflations = self._detect_pixel_inflations(
                eit_data,
                continuous_data,
                sequence,
                self.breath_detection_kwargs,
            )
            breath_data = pixel_inflations.values
        else:  # tiv_timing == "global"
            global_breaths = self._detect_breaths(
                continuous_data,
                eit_data.sample_frequency,
                self.breath_detection_kwargs,
            )
            breath_data = global_breaths.values

        number_of_breaths = (
            len(next(b for b in breath_data.flatten() if b)) if tiv_timing == "pixel" else len(breath_data)
        )
        tiv_values_array = np.empty((number_of_breaths, rows, cols), dtype=object)

        for i in range(number_of_breaths):
            for row in range(rows):
                for col in range(cols):
                    if tiv_timing == "pixel" and not breath_data[row, col]:
                        tiv_values_array[i, row, col] = np.nan
                    else:
                        time_series = data[:, row, col]
                        tiv_values = self._calculate_tiv_values(
                            time_series,
                            eit_data.time,
                            breath_data[row, col] if tiv_timing == "pixel" else breath_data,
                            tiv_method,
                        )
                        tiv_values_array[i, row, col] = tiv_values[i]

        return tiv_values_array.astype(float)
