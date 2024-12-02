from abc import ABC, abstractmethod

import numpy as np


class TimeDomainFilter(ABC):
    """Parent class for time domain filters."""

    available_in_gui = True

    @abstractmethod
    def apply_filter(self, input_data: np.ndarray) -> np.ndarray:
        """Apply the filter to the input data."""
        ...
