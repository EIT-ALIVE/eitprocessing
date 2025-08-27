from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData

T = TypeVar("T", bound=np.ndarray | ContinuousData | EITData)


class TimeDomainFilter(ABC):
    """Parent class for time domain filters."""

    available_in_gui = True

    @abstractmethod
    def apply(self, input_data: T, **kwargs) -> T:
        """Apply the filter to the input data."""
        ...
