from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Self

from eitprocessing.datahandling.mixins.equality import Equivalence


@dataclass(eq=False)
class DataContainer(Equivalence):
    """Base class for data container classes."""

    def deepcopy(self) -> Self:
        """Return a deep copy of the object."""
        return deepcopy(self)
