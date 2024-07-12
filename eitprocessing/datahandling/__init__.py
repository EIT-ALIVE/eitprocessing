from dataclasses import dataclass

from eitprocessing.datahandling.mixins.equality import Equivalence


@dataclass(eq=False)
class DataContainer(Equivalence):
    """Base class for data container classes."""
