from dataclasses import dataclass, field
from typing import Any

import numpy as np
from typing_extensions import Self

from eitprocessing.datahandling.mixins.slicing import SelectByTime


@dataclass
class SparseData(SelectByTime):
    """Container for sparse data.

    Sparse data does not have a set time between data points. Examples are data points at end of inspiration/end of
    expiration (e.g. tidal volume, end-expiratoy lung impedance) or detected time points (e.g. QRS complexes).

    Sparse data can consist of only time (e.g. detected QRS complexes) or time-value pairs (e.g. tidal impedance
    variation at the end of each breath).

    Values will generally be numeric values in arrays, but can also be lists of different types of object.

    Args:
        label: Computer readable name.
        name: Human readable name.
        unit: Unit of the data, if applicable.
        category: Category the data falls into, e.g. 'airway pressure'.
        description: Human readible extended description of the data.
        parameters: Parameters used to derive the data.
        derived_from: Traceback of intermediates from which the current data was derived.
        values: List or array of values. These van be numeric data, text or Python objects.
    """

    label: str
    name: str
    unit: str | None
    category: str
    time: np.ndarray | None
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    derived_from: list[Any] = field(default_factory=list)
    values: Any | None = None

    def __post_init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.label}')"

    def _sliced_copy(
        self,
        start_index: int,
        end_index: int,
        label: str,
    ) -> Self:
        # TODO: check correct implementation
        cls = self.__class__
        time = self.time[start_index:end_index]
        values = self.values[start_index:end_index] if self.values is not None else None
        description = f"Slice ({start_index}-{end_index}) of <{self.description}>"

        return cls(
            label=label,
            name=self.name,
            unit=self.unit,
            category=self.category,
            description=description,
            derived_from=[*self.derived_from, self],
            time=time,
            values=values,
        )
