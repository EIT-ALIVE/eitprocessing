from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from eitprocessing.datahandling.mixins.equality import Equivalence
from eitprocessing.datahandling.mixins.locking import Lockable
from eitprocessing.datahandling.mixins.slicing import SelectByTime

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Any, Self

T = TypeVar("T", bound="ContinuousData")


@dataclass(eq=False)
class ContinuousData(Equivalence, SelectByTime, Lockable):
    """Container for data with a continuous time axis.

    Continuous data is data that was continuously measured/created at a predictable rate. Therefore, continuous data is
    assumed to have a predictable delta between time points. A fixed delta is not enforced for two reasons: a) some
    devices have slightly varying sampling rate (but fixed around a set rate) and b) floating point arithmetic.

    Continuous data is assumed to be sequential (i.e. a single data point at each time point, sorted by time).

    Args:
        label: Computer readable naming of the instance.
        name: Human readable naming of the instance.
        unit: Unit of the data, if applicable.
        category: Category the data falls into, e.g. 'airway pressure'.
        description: Human readable extended description of the data.
        parameters: Parameters used to derive this data.
        derived_from: Traceback of intermediates from which the current data was derived.
        values: Data points.
    """  # TODO: update docstring

    label: str = field(compare=False)
    name: str = field(compare=False, repr=False)
    unit: str = field(metadata={"check_equivalence": True}, repr=False)
    category: str = field(metadata={"check_equivalence": True}, repr=False)
    description: str = field(default="", compare=False, repr=False)
    parameters: dict[str, Any] = field(default_factory=dict, repr=False, metadata={"check_equivalence": True})
    derived_from: Any | list[Any] = field(default_factory=list, repr=False, compare=False)
    time: np.ndarray = field(kw_only=True, repr=False, metadata={"lock_action_default": True})
    values: np.ndarray = field(kw_only=True, repr=False, metadata={"lock_action_default": True})

    def __post_init__(self) -> None:
        self.lock()

    def __setattr__(self, attr: str, value: Any):  # noqa: ANN401
        try:
            getattr(self, attr)
        except AttributeError:
            pass
        else:
            if self.is_lockable(attr) and self.is_locked(attr):
                msg = f"Attribute '{attr}' is locked and can't be overwritten."
                raise AttributeError(msg)
        super().__setattr__(attr, value)

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
            derived_from=[*self.derived_from, self],
            category=self.category,
            # copying data can become inefficient with large datasets if the
            # data is not directly edited afer copying but overridden instead;
            # consider creating a view and locking it, requiring the user to
            # make a copy if they want to edit the data directly
            time=np.copy(self.time),
            values=np.copy(self.values),
        )
        obj.unlock_all()
        return obj

    def __add__(self: T, other: T) -> T:
        return self.concatenate(other)

    def concatenate(self: T, other: T, newlabel: str | None = None) -> T:  # noqa: D102, will be removed soon
        # TODO: compare both concatenate methods and check what is needed from both and merge into one
        # Check that data can be concatenated
        self.isequivalent(other, raise_=True)
        if np.min(other.time) <= np.max(self.time):
            msg = f"{other} (b) starts before {self} (a) ends."
            raise ValueError(msg)

        cls = self.__class__
        newlabel = newlabel or self.label

        return cls(
            name=self.name,
            label=self.label,  # TODO: using newlabel leads to errors
            unit=self.unit,
            category=self.category,
            time=np.concatenate((self.time, other.time)),
            values=np.concatenate((self.values, other.values)),
            derived_from=[*self.derived_from, *other.derived_from, self, other],
        )

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

    @property
    def loaded(self) -> bool:
        """Return whether the data was loaded from disk, or derived from elsewhere."""
        return len(self.derived_from) == 0

    def __len__(self):
        return len(self.time)

    def _sliced_copy(
        self,
        start_index: int,
        end_index: int,
        newlabel: str,  # noqa: ARG002
    ) -> Self:
        # TODO: check correct implementation
        cls = self.__class__
        time = self.time[start_index:end_index]
        values = self.values[start_index:end_index]
        description = f"Slice ({start_index}-{end_index}) of <{self.description}>"

        return cls(
            label=self.label,  # TODO: newlabel gives errors
            name=self.name,
            unit=self.unit,
            category=self.category,
            description=description,
            derived_from=[*self.derived_from, self],
            time=time,
            values=values,
        )
