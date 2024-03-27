from dataclasses import dataclass, field
from typing import Any

import numpy as np
from typing_extensions import Self

from eitprocessing.datahandling.mixins.slicing import SelectByTime


@dataclass
class SparseData(SelectByTime):
    """Container for single value or sparse data.

    Sparse data is data that does not appear/change at predictable regular time intervals. Examples are detected breaths
    or heart beats, tidal impedance variations per breath, or a single PEEP value for a PEEP step during a decremental
    PEEP trial.

    Sparse data can consist of only time (e.g. detected QRS complexes), time-value pairs (e.g. tidal impedance variation
    of breaths at the end of each breath), values without time (e.g. objects) or a single value (e.g. the PEEP level of
    the current PEEP step).

    Either the value or values can be provided, but not both.

    Args:
        label: Computer readable name.
        name: Human readable name.
        unit: Unit of the data, if applicable.
        category: Category the data falls into, e.g. 'airway pressure'.
        description: Human readible extended description of the data.
        parameters: Parameters used to derive the data.
        derived_from: Traceback of intermediates from which the current data was derived.
        values: List or array of values. These van be numeric data, text or Python objects.
        value: single value. This van be numeric, text or a Python object.
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
