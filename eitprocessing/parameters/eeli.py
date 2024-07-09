from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.features.breath_detection import BreathDetection
from eitprocessing.parameters import ParameterCalculation


@dataclass
class EELI(ParameterCalculation):
    """Compute the end-expiratory lung impedance (EELI) per breath."""

    method: Literal["breath_detection"] = "breath_detection"
    breath_detection_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        pass

    def compute_parameter(self, continuous_data: ContinuousData, sample_frequency: float) -> np.ndarray:
        """Compute the EELI per breath.

        Args:
            continuous_data: a ContinuousData object containing the data.
            sample_frequency: the sample frequency at which the data is recorded.
        """
        # TODO: remove sample_frequency as soon as ContinuousData gets it as attribute

        if self.method != "breath_detection":
            msg = f"Method {self.method} is not implemented."
            raise NotImplementedError(msg)

        data = continuous_data.values

        bd_kwargs = self.breath_detection_kwargs.copy()
        bd_kwargs["sample_frequency"] = sample_frequency
        breath_detection = BreathDetection(**bd_kwargs)
        breaths = breath_detection.find_breaths(data)

        _, _, end_expiratory_times = zip(*breaths, strict=True)
        end_expiratory_indices = np.flatnonzero(np.isin(continuous_data.time, end_expiratory_times))

        return data[end_expiratory_indices]
