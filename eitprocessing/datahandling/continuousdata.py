from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from eitprocessing.datahandling import DataContainer
from eitprocessing.datahandling.mixins.slicing import SelectByTime

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Any, Self

T = TypeVar("T", bound="ContinuousData")


@dataclass(eq=False)
class ContinuousData(DataContainer, SelectByTime):
    """Container for data with a continuous time axis.

    Continuous data is assumed to be sequential (i.e. a single data point at each time point, sorted by time) and
    continuously measured/created at a fixed sampling frequency. However, a fixed interval between consecutive time
    points is not enforced to account for floating point arithmetic, devices with imperfect sampling frequencies, and
    other sources of variation.

    Args:
        label: Computer readable naming of the instance.
        name: Human readable naming of the instance.
        unit: Unit of the data, if applicable.
        category: Category the data falls into, e.g. 'airway pressure'.
        description: Human readable extended description of the data.
        parameters: Parameters used to derive this data.
        derived_from: Traceback of intermediates from which the current data was derived.
        values: Data points.
    """

    label: str = field(compare=False)
    name: str = field(compare=False, repr=False)
    unit: str = field(metadata={"check_equivalence": True}, repr=False)
    category: str = field(metadata={"check_equivalence": True}, repr=False)
    description: str = field(default="", compare=False, repr=False)
    parameters: dict[str, Any] = field(default_factory=dict, repr=False, metadata={"check_equivalence": True})
    derived_from: Any | list[Any] = field(default_factory=list, repr=False, compare=False)
    time: np.ndarray = field(kw_only=True, repr=False)
    values: np.ndarray = field(kw_only=True, repr=False)
    sample_frequency: float | None = field(kw_only=True, repr=False, metadata={"check_equivalence": True}, default=None)

    def __post_init__(self) -> None:
        if self.loaded:
            self.lock()
        self.lock("time")

        if self.sample_frequency is None:
            msg = (
                "`sample_frequency` is set to `None`. This will not be supported in future versions. "
                "Provide a sample frequency when creating a ContinuousData object."
            )
            warnings.warn(msg, DeprecationWarning)

        if (lv := len(self.values)) != (lt := len(self.time)):
            msg = f"The number of time points ({lt}) does not match the number of values ({lv})."
            raise ValueError(msg)

    def __setattr__(self, attr: str, value: Any):  # noqa: ANN401
        try:
            old_value = getattr(self, attr)
        except AttributeError:
            pass
        else:
            if isinstance(old_value, np.ndarray) and old_value.flags["WRITEABLE"] is False:
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
            sample_frequency=self.sample_frequency,
        )
        obj.unlock()
        return obj

    def __add__(self: Self, other: Self) -> Self:
        return self.concatenate(other)

    def concatenate(self: Self, other: Self, newlabel: str | None = None) -> Self:  # noqa: D102, will be removed soon
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
            sample_frequency=self.sample_frequency,
        )

    def derive(
        self,
        label: str,
        function: Callable,
        func_args: dict | None = None,
        **kwargs,
    ) -> Self:
        """Create a copy deriving data from values attribute.

        Args:
            label: New label for the derived object.
            function: Function that takes the values and returns the derived values.
            func_args: Arguments to pass to function, if any.
            **kwargs: Values for changed attributes of derived object.

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
        if func_args is None:
            func_args = {}
        copy = self.copy(label, **kwargs)
        copy.values = function(copy.values, **func_args)
        return copy

    def lock(self, *attr: str) -> None:
        """Lock attributes, essentially rendering them read-only.

        Locked attributes cannot be overwritten. Attributes can be unlocked using `unlock()`.

        Args:
            *attr: any number of attributes can be passed here, all of which will be locked. Defaults to "values".

        Examples:
            >>> # lock the `values` attribute of `data`
            >>> data.lock()
            >>> data.values = [1, 2, 3] # will result in an AttributeError
            >>> data.values[0] = 1      # will result in a RuntimeError
        """
        if not attr:
            # default values are not allowed when using *attr, so set a default here if none is supplied
            attr = ("values",)
        for attr_ in attr:
            getattr(self, attr_).flags["WRITEABLE"] = False

    def unlock(self, *attr: str) -> None:
        """Unlock attributes, rendering them editable.

        Locked attributes cannot be overwritten, but can be unlocked with this function to make them editable.

        Args:
            *attr: any number of attributes can be passed here, all of which will be unlocked. Defaults to "values".

        Examples:
            >>> # lock the `values` attribute of `data`
            >>> data.lock()
            >>> data.values = [1, 2, 3] # will result in an AttributeError
            >>> data.values[0] = 1      # will result in a RuntimeError
            >>> data.unlock()
            >>> data.values = [1, 2, 3]
            >>> print(data.values)
            [1,2,3]
            >>> data.values[0] = 1      # will result in a RuntimeError
            >>> print(data.values)
            1
        """
        if not attr:
            # default values are not allowed when using *attr, so set a default here if none is supplied
            attr = ("values",)
        for attr_ in attr:
            getattr(self, attr_).flags["WRITEABLE"] = True

    @property
    def locked(self) -> bool:
        """Return whether the values attribute is locked.

        See lock().
        """
        return not self.values.flags["WRITEABLE"]

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
        time = np.copy(self.time[start_index:end_index])
        values = np.copy(self.values[start_index:end_index])
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
            sample_frequency=self.sample_frequency,
        )
