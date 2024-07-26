from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np

from eitprocessing.categories import check_category
from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.features.breath_detection import BreathDetection
from eitprocessing.parameters import ParameterCalculation


@dataclass
class EELI(ParameterCalculation):
    """Compute the end-expiratory lung impedance (EELI) per breath."""

    method: Literal["breath_detection"] = "breath_detection"
    breath_detection_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        _methods = get_args(EELI.__dataclass_fields__["method"].type)
        if self.method not in _methods:
            msg = f"Method {self.method} is not valid. Use any of {', '.join(_methods)}"
            raise ValueError(msg)

    def compute_parameter(self, continuous_data: ContinuousData) -> np.ndarray:
        """Compute the EELI for each breath in the impedance data.

        Example:
        >>> global_impedance = sequence.continuous_data["global_impedance_(raw)"]
        >>> eeli_values = EELI().compute_parameter(global_impedance)

        Args:
            continuous_data: a ContinuousData object containing impedance data.

        Returns:
            np.ndarray: the end-expiratory values of all breaths in the impedance data.
        """
        # TODO: remove sample_frequency as soon as ContinuousData gets it as attribute

        check_category(continuous_data, "impedance", raise_=True)

        bd_kwargs = self.breath_detection_kwargs.copy()
        breath_detection = BreathDetection(**bd_kwargs)
        breaths = breath_detection.find_breaths(continuous_data)

        if not len(breaths):
            return np.array([], dtype=float)

        _, _, end_expiratory_times = zip(*breaths.values, strict=True)
        end_expiratory_indices = np.flatnonzero(np.isin(continuous_data.time, end_expiratory_times))

        return continuous_data.values[end_expiratory_indices]
