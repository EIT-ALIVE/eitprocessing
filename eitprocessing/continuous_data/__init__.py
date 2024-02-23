from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Any, Self


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

    def __post_init__(self) -> None:
        if not self.loaded and not self.derived_from:
            msg = "Data must be loaded or calculated form another dataset."
            raise ValueError(msg)

    def copy(
        self,
        label: str,
        *,
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        parameters: dict | None = None,
    ) -> Self:
        return self.__class__(
            label=label,
            name=name or label,
            unit=unit or self.unit,
            description=description or f"Derived from {self.name}",
            parameters=self.parameters | (parameters or {}),
            loaded=False,
            derived_from=[*self.derived_from, self],
            category=self.category,
            # copying data can become inefficient with large datasets if the
            # data is not directly edited afer copying but overridden instead;
            # consider creating a view and locking it, requiring the user to
            # make a copy if they want to edit the data directly
            values=np.copy(self.values),
        )

    def derive(self, label: str, function: Callable, **kwargs) -> Self:
        copy = self.copy(label)
        copy.values = function(copy.values, **kwargs)
        return copy

    def lock(self) -> None:
        self.values.flags["WRITEABLE"] = False

    def unlock(self) -> None:
        self.values.flags["WRITEABLE"] = True

    @property
    def locked(self) -> bool:
        return not self.values.flags["WRITEABLE"]
