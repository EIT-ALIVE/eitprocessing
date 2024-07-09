from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.breath_detection import BreathDetection
from eitprocessing.parameters import ParameterExtraction


@dataclass
class EELI(ParameterExtraction):
    """Compute the end-expiratory lung impedance (EELI) per breath."""

    method: Literal["extremes"] = "extremes"
    breath_detection_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        pass

    def compute_parameter(self, sequence: Sequence, data_label: str) -> dict | list[dict]:
        """Compute the EELI per breath.

        Args:
            sequence: the sequence containing the data.
            data_label: the label of the continuous data in the sequence to determine the EELI of.
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

        _, _, eeli_indices = zip(*breaths, strict=True)
        eeli_indices = list(eeli_indices)
        eeli_values = data[eeli_indices]

        return eeli_values
