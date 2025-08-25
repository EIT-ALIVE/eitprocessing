from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from eitprocessing.datahandling.eitdata import EITData
from tests.test_breath_detection import ContinuousData

T = TypeVar("T", bound=np.ndarray | ContinuousData | EITData)


class TimeDomainFilter(ABC):
    """Parent class for time domain filters."""

    available_in_gui = True

    @abstractmethod
    def apply(self, input_data: T, **kwargs) -> T:
        """Apply the filter to the input data."""
        ...
