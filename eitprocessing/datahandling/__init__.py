import dataclasses
from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Self

from eitprocessing.datahandling.mixins.equality import Equivalence


@dataclass(eq=False)
class DataContainer(Equivalence):
    """Base class for data container classes."""

    def __bool__(self):
        return True

    def deepcopy(self) -> Self:
        """Return a deep copy of the object."""
        return deepcopy(self)


@dataclass(eq=False, frozen=True)
class FrozenDataContainer(Equivalence):
    """Base class for data container classes."""

    def __bool__(self):
        return True

    def deepcopy(self) -> Self:
        """Return a deep copy of the object."""
        return deepcopy(self)

    def update(self: Self, **kwargs: object) -> Self:
        """Return a copy of the object with specified fields replaced.

        Args:
            **kwargs: Fields to replace.

        Returns:
            A new instance of the object with the specified fields replaced.
        """
        return dataclasses.replace(self, **kwargs)
