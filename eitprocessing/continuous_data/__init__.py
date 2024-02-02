from dataclasses import dataclass
from dataclasses import field
import numpy as np
from typing_extensions import Any
from typing_extensions import Self


@dataclass
class ContinuousData:
    label: str
    name: str
    unit: str
    category: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    loaded: bool = False
    derived_from: Any | list[Any] = field(default_factory=list)
    values: np.ndarray = field(kw_only=True)

    def __post_init__(self):
        if not self.loaded and not self.derived_from:
            raise ValueError("Data must be loaded or calculated form another dataset.")

    def copy(
        self, label, *, name=None, unit=None, description=None, parameters=None
    ) -> Self:
        return self.__class__(
            label=label,
            name=name or label,
            unit=unit or self.unit,
            description=description or f"Derived from {self.name}",
            parameters=self.parameters | (parameters or {}),
            loaded=False,
            derived_from=self.derived_from + [self],
            category=self.category,
            # copying data can become inefficient with large datasets if the
            # data is not directly edited afer copying but overridden instead;
            # consider creating a view and locking it, requiring the user to
            # make a copy if they want to edit the data directly
            values=np.copy(self.values),
        )

    def derive(self, label, function, **kwargs) -> Self:
        copy = self.copy(label)
        copy.values = function(copy.values)
        return copy
