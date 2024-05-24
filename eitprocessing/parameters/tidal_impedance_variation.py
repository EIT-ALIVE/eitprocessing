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
    summary_stats: dict = field(
        default_factory=lambda: {
            "values": lambda v: v,
            "mean": np.mean,
            "standard deviation": np.std,
            "median": np.median,
        },
    )
    breath_detection_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        pass

    def compute_parameter(self, sequence: Sequence, data_label: str, tiv_method = 'inspiratory') -> dict | list[dict]:
        """Compute the tidal impedance variation per breath.

        Args:
            sequence: the sequence containing the data.
            data_label: the label of the continuous data in the sequence to determine the TIV of.
            breaths_label: the label of the breaths in the sequence.
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
        breaths = breath_detection.find_breaths(data)

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

        if tiv_method == 'triangular':
            msg = f"Method {tiv_method} is not implemented."
            raise NotImplementedError(msg)

        result = {}
        for name, function in self.summary_stats.items():
            result[name] = function(tiv_values)

        result["peak indices"] = middle_indices

        return result
