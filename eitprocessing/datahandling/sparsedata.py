import copy
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
from typing_extensions import Self

from eitprocessing.datahandling import DataContainer
from eitprocessing.datahandling.mixins.slicing import SelectByTime

T = TypeVar("T", bound="SparseData")


@dataclass(eq=False)
class SparseData(DataContainer, SelectByTime):
    """Container for data related to individual time points.

    Sparse data is data for which the time points are not necessarily evenly spaced. Data can consist time-value pairs
    or only time points.

    Sparse data differs from interval data in that each data points is associated with a single time point rather than a
    time range.

    Examples are data points at end of inspiration/end of expiration (e.g. tidal volume, end-expiratoy lung impedance)
    or detected time points (e.g. QRS complexes).

    Args:
        label: Computer readable name.
        name: Human readable name.
        unit: Unit of the data, if applicable.
        category: Category the data falls into, e.g. 'detected r peak'.
        description: Human readable extended description of the data.
        parameters: Parameters used to derive the data.
        derived_from: Traceback of intermediates from which the current data was derived.
        values: List or array of values. These van be numeric data, text or Python objects.
    """

    label: str = field(compare=False)
    name: str = field(compare=False, repr=False)
    unit: str | None = field(metadata={"check_equivalence": True}, repr=False)
    category: str = field(metadata={"check_equivalence": True}, repr=False)
    time: np.ndarray = field(repr=False)
    description: str = field(compare=False, default="", repr=False)
    parameters: dict[str, Any] = field(default_factory=dict, metadata={"check_equivalence": True}, repr=False)
    derived_from: list[Any] = field(default_factory=list, compare=False, repr=False)
    values: Any | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.label}')"

    def __len__(self) -> int:
        return len(self.time)

    def __post_init__(self):
        if self.has_values and (lv := len(self.values)) != (lt := len(self.time)):
            msg = f"The number of time points ({lt}) does not match the number of values ({lv})."
            raise ValueError(msg)

    @property
    def has_values(self) -> bool:
        """True if the SparseData has values, False otherwise."""
        return self.values is not None

    def _sliced_copy(
        self,
        start_index: int,
        end_index: int,
        newlabel: str,
    ) -> Self:
        cls = self.__class__
        time = self.time[start_index:end_index]
        values = self.values[start_index:end_index] if self.has_values else None
        description = f"Slice ({start_index}-{end_index}) of <{self.description}>"

        return cls(
            label=newlabel,
            name=self.name,
            unit=self.unit,
            category=self.category,
            description=description,
            derived_from=[*self.derived_from, self],
            time=time,
            values=values,
        )

    def __add__(self: Self, other: Self) -> Self:
        return self.concatenate(other)

    def concatenate(self, other: Self, newlabel: str | None = None) -> Self:  # noqa: D102, will be moved to mixin in future
        self.isequivalent(other, raise_=True)

        # TODO: make proper copy functions
        if not len(self):
            return copy.deepcopy(other)
        if not len(other):
            return copy.deepcopy(self)

        if np.min(other.time) <= np.max(self.time):
            msg = f"{other} (b) starts before {self} (a) ends."
            raise ValueError(msg)

        cls = type(self)
        newlabel = newlabel or self.label

        if type(self.values) is not type(other.values):
            msg = "Concatenation failed because value types are non-identical."
            raise TypeError(msg)
        if not self.has_values:
            new_values = None
        elif isinstance(self.values, np.ndarray):
            new_values = np.concatenate((self.values, other.values))
        else:
            try:
                new_values = self.values + other.values
            except TypeError as e:
                msg = "Concatenation failed because values could not be concatenated."
                raise TypeError(msg) from e

        return cls(
            label=newlabel,
            name=self.name,
            unit=self.unit,
            category=self.category,
            description=self.description,
            derived_from=[*self.derived_from, *other.derived_from, self, other],
            time=np.concatenate((self.time, other.time)),
            values=new_values,
        )
