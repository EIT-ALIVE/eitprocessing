from abc import ABC, abstractmethod

import numpy as np

from eitprocessing.datahandling.continuousdata import ContinuousData


class ParameterCalculation(ABC):
    """Base class for parameter extraction classes."""

    @abstractmethod
    def compute_parameter(self, input_data: ContinuousData) -> np.ndarray:
        """Computes the parameter."""
