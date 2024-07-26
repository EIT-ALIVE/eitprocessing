from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.breath_detection import BreathDetection
from eitprocessing.features.pixel_inflation import PixelInflation
from eitprocessing.parameters import ParameterExtraction


@dataclass
class TIV(ParameterExtraction):
    """Compute the tidal impedance variation (TIV) per breath."""

    method: str = "extremes"
    summary_stats_global: dict = field(
        default_factory=lambda: {
            "values": lambda v: v,
            "mean": np.mean,
            "standard deviation": np.std,
            "median": np.median,
        },
    )
    summary_stats_pixel: dict = field(
        default_factory=lambda: {
            "values": lambda v: v,
            "mean": lambda v: np.mean(v, axis=0),
            "standard deviation": lambda v: np.std(v, axis=0),
            "median": lambda v: np.median(v, axis=0),
        },
    )
    breath_detection_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        pass

    def _detect_breaths(self, data, framerate, breath_detection_kwargs):
        bd_kwargs = breath_detection_kwargs.copy()
        bd_kwargs["sample_frequency"] = framerate
        breath_detection = BreathDetection(**bd_kwargs)
        return breath_detection.find_breaths(data)

    def _detect_pixel_inflations(self, sequence, framerate, breath_detection_kwargs):
        bd_kwargs = breath_detection_kwargs.copy()
        bd_kwargs["sample_frequency"] = framerate
        pi = PixelInflation(**bd_kwargs)
        return pi.find_pixel_inflations(
            sequence,
            eitdata_label="raw",
            continuousdata_label="global_impedance_(raw)",
        )

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

    def compute_global_parameter(
        self,
        sequence: Sequence,
        data_label: str,
        tiv_method="inspiratory",
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

        continuousdata = sequence.continuous_data[data_label]
        eitdata = next(filter(lambda x: isinstance(x, EITData), continuousdata.derived_from))

        breaths = self._detect_breaths(continuousdata, eitdata.framerate, self.breath_detection_kwargs)
        tiv_values = self._calculate_tiv_values(continuousdata.values, continuousdata.time, breaths.values, tiv_method)

        result = {name: function(tiv_values) for name, function in self.summary_stats_global.items()}
        result["peak indices"] = [middle for _, middle, _ in breaths.values]

        return result

    def compute_pixel_parameter(
        self,
        sequence: Sequence,
        data_label: str,
        tiv_method: Literal["inspiratory", "expiratory", "mean"] = "inspiratory",
        tiv_timing: Literal["pixel", "global"] = "pixel",
        continuous_data_label: str | None = None,
    ) -> dict | list[dict]:
        """Compute the tidal impedance variation per breath on pixel level.

        Args:
            sequence: The sequence containing the data.
            data_label: The label of the eit data in the sequence to determine the TIV of.
            tiv_method: The label of which part of the breath the TIV should be determined on
                        (inspiratory, expiratory or mean). Defaults to 'inspiratory'.
            tiv_timing: The label of which timing should be used to compute the TIV,
                        either based on the global breaths ('global') or pixel inflations ('pixel').
                        Defaults to 'pixel'.
            continuous_data_label: The label of the continuous data in the sequence to determine
                                the TIV of, required if tiv_timing is 'global'.

        Raises:
            NotImplementedError: If the method is not 'extremes'.
            ValueError: If tiv_method is not one of 'inspiratory', 'expiratory', or 'mean'.
            ValueError: If tiv_timing is not one of 'global' or 'pixel'.
            ValueError: If tiv_timing is 'global' and continuous_data_label is not provided.
        """
        # TODO: think about other name for 'global', since it can also be regional, functional, etc.
        if self.method != "extremes":
            msg = f"Method {self.method} is not implemented."
            raise NotImplementedError(msg)

        if tiv_method not in ["inspiratory", "expiratory", "mean"]:
            msg = "tiv_method must be either 'inspiratory', 'expiratory' or 'mean'"
            raise ValueError(msg)

        if tiv_timing not in ["global", "pixel"]:
            msg = "tiv_timing must be either 'global' or 'pixel'"
            raise ValueError(msg)

        if tiv_timing == "global" and not continuous_data_label:
            msg = "continuous_data_label must be provided when tiv_timing is 'global'"
            raise ValueError(msg)

        data = sequence.eit_data[data_label].pixel_impedance
        _, rows, cols = data.shape

        if tiv_timing == "pixel":
            pixel_inflations = self._detect_pixel_inflations(
                sequence,
                sequence.eit_data[data_label].framerate,
                self.breath_detection_kwargs,
            )
            breath_data = pixel_inflations.values
        else:  # tiv_timing == "global"
            global_breaths = self._detect_breaths(
                sequence.continuous_data[continuous_data_label],
                sequence.eit_data[data_label].framerate,
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
                            sequence.eit_data[data_label].time,
                            breath_data[row, col] if tiv_timing == "pixel" else breath_data,
                            tiv_method,
                        )
                        tiv_values_array[i, row, col] = tiv_values[i]

        tiv_values_array = tiv_values_array.astype(float)
        result = {name: function(tiv_values_array) for name, function in self.summary_stats_pixel.items()}

        return result
