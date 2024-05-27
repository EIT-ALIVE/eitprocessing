from dataclasses import dataclass, field

import numpy as np

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.breath_detection import BreathDetection
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


    def compute_global_parameter(self, sequence: Sequence, data_label: str, tiv_method = 'inspiratory') -> dict | list[dict]:
        """Compute the tidal impedance variation per breath.

        Args:
            sequence: the sequence containing the data.
            data_label: the label of the continuous data in the sequence to determine the TIV of.
            tiv_method: the label of which part of the breath the TIV should be determined on (inspiratory, expiratory or mean)
        """
        if self.method != "extremes":
            msg = f"Method {self.method} is not implemented."
            raise NotImplementedError(msg)

        continuousdata = sequence.continuous_data[data_label]
        eitdata = next(filter(lambda x: isinstance(x, EITData), continuousdata.derived_from))
        data = continuousdata.values

        bd_kwargs = self.breath_detection_kwargs.copy()
        bd_kwargs["sample_frequency"] = eitdata.framerate
        breath_detection = BreathDetection(**bd_kwargs)
        breaths = breath_detection.find_breaths_1d(data)

        start_indices, middle_indices, end_indices = zip(*breaths, strict=True)
        start_indices = list(start_indices)
        middle_indices = list(middle_indices)
        end_indices = list(end_indices)

        if tiv_method == 'inspiratory':
            end_inspiratory_values = data[middle_indices]
            start_inspiratory_values = data[start_indices]
            tiv_values = end_inspiratory_values - start_inspiratory_values

        if tiv_method == 'expiratory':
            start_expiratory_values = data[middle_indices]
            end_expiratory_values = data[end_indices]
            tiv_values = start_expiratory_values - end_expiratory_values

        if tiv_method == 'mean':
            start_inspiratory_values = data[start_indices]
            end_inspiratory_values = data[middle_indices]
            end_expiratory_values = data[end_indices]
            tiv_values = end_inspiratory_values - [np.mean(k) for k in zip(start_inspiratory_values, end_expiratory_values)]

        result = {}
        for name, function in self.summary_stats_global.items():
            result[name] = function(tiv_values)

        result["peak indices"] = middle_indices

        return result


    def compute_pixel_parameter(self, sequence: Sequence, data_label: str, tiv_method = 'inspiratory') -> dict | list[dict]:
        """Compute the tidal impedance variation per breath on pixel level.

        Args:
            sequence: the sequence containing the data.
            data_label: the label of the continuous data in the sequence to determine the TIV of.
            tiv_method: the label of which part of the breath the TIV should be determined on (inspiratory, expiratory or mean)
        """
        if self.method != "extremes":
            msg = f"Method {self.method} is not implemented."
            raise NotImplementedError(msg)

        data = sequence.eit_data[data_label].pixel_impedance

        bd_kwargs = self.breath_detection_kwargs.copy()
        bd_kwargs["sample_frequency"] = sequence.eit_data[data_label].framerate
        breath_detection = BreathDetection(**bd_kwargs)
        breaths = breath_detection.find_breaths_2d(data)

        rows, cols = breaths.shape
        tiv_values_array = np.empty((11, rows, cols), dtype=object)

        # Create a boolean mask indicating non-empty lists
        non_empty_mask = np.array([[bool(lst) for lst in row] for row in breaths])

        # Get the indices of non-empty lists
        non_empty_index = np.argwhere(non_empty_mask)[0]
        number_of_breaths = len(breaths[non_empty_index[0], non_empty_index[1]])

        for i in range(number_of_breaths):
            for row in range(rows):
                for col in range(cols):
                    if not breaths[row, col]:
                        tiv_values_array[i, row, col] = np.nan
                        continue

                    time_series = data[:, row, col]
                    start_indices, middle_indices, end_indices = zip(
                        *breaths[row, col], strict=True
                    )
                    start_indices = list(start_indices)
                    middle_indices = list(middle_indices)
                    end_indices = list(end_indices)

                    if tiv_method == "inspiratory":
                        end_inspiratory_values = time_series[middle_indices]
                        start_inspiratory_values = time_series[start_indices]
                        tiv_values = end_inspiratory_values - start_inspiratory_values

                    if tiv_method == "expiratory":
                        start_expiratory_values = time_series[middle_indices]
                        end_expiratory_values = time_series[end_indices]
                        tiv_values = start_expiratory_values - end_expiratory_values

                    if tiv_method == "mean":
                        start_inspiratory_values = time_series[start_indices]
                        end_inspiratory_values = time_series[middle_indices]
                        end_expiratory_values = time_series[end_indices]
                        tiv_values = end_inspiratory_values - [
                            np.mean(k) for k in zip(start_inspiratory_values, end_expiratory_values)
                        ]

                    tiv_values_array[i, row, col] = tiv_values[i]

        tiv_values_array = tiv_values_array.astype(float)
        result = {}
        for name, function in self.summary_stats_pixel.items():
            result[name] = function(tiv_values_array)

        return result