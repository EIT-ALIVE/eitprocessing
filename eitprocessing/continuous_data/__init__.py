from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Any, Self


@dataclass
class ContinuousData:
    """Data class for (non-EIT) data with a continuous time axis.

    Args:
        label: Computer readable naming of the instance.
        name: Human readable naming of the instance.
        unit: Unit for the data.
        category: Category the data falls into, e.g. 'airway pressure'.
        description: Human readible extended description of the data.
        parameters: Parameters used to derive this data.
        loaded: True if raw data was loaded directly from source. False if the data was derived.
        derived_from: Traceback of intermediates from which the current data was derived.
        values: Data points.
    """

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

        if self.loaded:
            self.lock()

    def __setattr__(self, attr: str, value: Any):  # noqa: ANN401
        if attr == "values" and self.locked:
            msg = "Attribute 'values' is locked and can't be overwritten."
            raise AttributeError(msg)
        super().__setattr__(self, attr, value)

    def copy(
        self,
        label: str,
        *,
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        parameters: dict | None = None,
    ) -> Self:
        """Create a copy.

        Whenever data is altered, it should probably be copied first. The alterations should then be made in the copy.
        """
        obj = self.__class__(
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
        obj.unlock()
        return obj

    def derive(self, label: str, function: Callable, func_args: dict, **kwargs) -> Self:
        """Create a copy deriving data from values attribute.

        Args:
            label: New label for the derived object.
            function: Function that takes the values and returns the derived values.
            func_args: Arguments to pass to function.
            **kwargs: New values for attributes of

        Example:
        ```
        def convert_data(x, add=None, subtract=None, multiply=None, divide=None):
            if add:
                x += add
            if subtract:
                x -= subtract
            if multiply:
                x *= multiply
            if divide:
                x /= divide
            return x


        data = ContinuousData(
            name="Lung volume (in mL)", label="volume_mL", unit="mL", category="volume", values=some_loaded_data
        )
        derived = data.derive("volume_L", convert_data, {"divide": 1000}, name="Lung volume (in L)", unit="L")
        ```
        """
        copy = self.copy(label, **kwargs)
        copy.values = function(copy.values, **func_args)
        return copy

    def lock(self) -> None:
        """Lock the values attribute.

        When the values attribute is locked, it cannot be replaced or changed.
        `data.values = [1, 2, 3]` will result in an AttributeError being raised.
        `data.values[0] = 1` will result in a RuntimeError being raised.

        The values can be unlocked using `unlock()`.
        """
        self.values.flags["WRITEABLE"] = False

    def unlock(self) -> None:
        """Unlocks the values attribute.

        See lock().
        """
        self.values.flags["WRITEABLE"] = True

    @property
    def locked(self) -> bool:
        """Return whether the values attribute is locked.

        See lock().
        """
        return not self.values.flags["WRITEABLE"]
