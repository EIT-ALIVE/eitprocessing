from abc import ABC, abstractmethod
from typing import NoReturn

import numpy.typing as npt


class TimeDomainFilter(ABC):
    """Parent class for time domain filters."""

    available_in_gui = True

    @abstractmethod
    def apply_filter(self, input_data: npt.ArrayLike) -> NoReturn:
        """Apply the filter to the input data."""
        ...
