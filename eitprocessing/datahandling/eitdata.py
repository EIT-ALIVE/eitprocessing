from __future__ import annotations

import warnings
from dataclasses import KW_ONLY, InitVar, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from eitprocessing.datahandling import FrozenDataContainer
from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.mixins.slicing import SelectByTime
from eitprocessing.utils.frozen_array import freeze_array

if TYPE_CHECKING:
    from typing_extensions import Self


T = TypeVar("T", bound="EITData")


@dataclass(eq=False, frozen=True)
class EITData(FrozenDataContainer, SelectByTime):
    """Container for EIT impedance data.

    This class holds the pixel impedance from an EIT measurement, as well as metadata describing the measurement. The
    class is meant to hold data from (part of) a singular continuous measurement.

    This class can't be initialized directly. Instead, use `load_eit_data(<path>, vendor=<vendor>)` to load data from
    disk.

    Args:
        time: The time of each frame (since start measurement).
        values: Impedance values for each pixel at each frame.
        sample_frequency: The (average) frequency at which the frames are collected, in Hz.
        vendor: The vendor of the device the data was collected with.
        path: The path of list of paths of the source from which data was derived.
        label: Computer readable label identifying this dataset.
        name: Human readable name for the data.
        description: Human readable description of the data.
    """  # TODO: fix docstring

    time: np.ndarray = field(repr=False)
    values: np.ndarray = field(repr=False)
    _: KW_ONLY
    sample_frequency: float = field(metadata={"check_equivalence": True}, repr=False)
    vendor: Vendor = field(metadata={"check_equivalence": True}, repr=False)
    path: str | Path | list[Path | str] | None = field(compare=False, repr=False, default=None)
    label: str | None = field(default=None, compare=False, metadata={"check_equivalence": True})
    description: str | None = field(default=None, compare=False, repr=False)
    name: str | None = field(default=None, compare=False, repr=False)
    suppress_simulated_warning: InitVar[bool] = False

    def __init__(
        self,
        time: np.ndarray,
        values: np.ndarray | None = None,
        *,
        sample_frequency: float,
        vendor: Vendor | str,
        path: str | Path | list[Path | str] | None = None,
        label: str | None = None,
        description: str | None = None,
        name: str | None = None,
        suppress_simulated_warning: bool = False,
        **kwargs,
    ):
        values = self._parse_kwargs(values, kwargs)

        if not isinstance(values, np.ndarray):
            msg = f"'values' must be a numpy ndarray, not {type(values)}."
            raise TypeError(msg)

        label = label or f"{self.__class__.__name__}_{id(self)}"
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)

        if path is None:
            object.__setattr__(self, "path", None)
        else:
            path_list = self.ensure_path_list(path)
            if len(path_list) == 1:
                object.__setattr__(self, "path", path_list[0])
            else:
                object.__setattr__(self, "path", path_list)

        object.__setattr__(self, "sample_frequency", float(sample_frequency))
        if self.sample_frequency != sample_frequency:
            msg = (
                "Sample frequency could not be correctly converted from "
                f"{sample_frequency} ({type(sample_frequency)}) to "
                f"{self.sample_frequency:.1f} (float)."
            )
            raise TypeError(msg)

        if (lv := len(values)) != (lt := len(time)):
            msg = f"The number of time points ({lt}) does not match the number of pixel impedance values ({lv})."
            raise ValueError(msg)

        object.__setattr__(self, "values", freeze_array(values))
        object.__setattr__(self, "time", freeze_array(time))

        vendor = Vendor(vendor)
        if not suppress_simulated_warning and vendor == Vendor.SIMULATED:
            warnings.warn(
                "The simulated vendor is used for testing purposes. "
                "It is not a real vendor and should not be used in production code.",
                UserWarning,
                stacklevel=2,
            )
        object.__setattr__(self, "vendor", vendor)

    def _parse_kwargs(self, values: np.ndarray | None, kwargs: dict[str, Any]) -> np.ndarray | None:
        if "pixel_impedance" in kwargs:
            if values is not None:
                msg = "Cannot provide both 'pixel_impedance' and 'values'."
                raise ValueError(msg)
            warnings.warn("`pixel_impedance` has been replaced by `values`.", DeprecationWarning, stacklevel=2)
            values = kwargs.pop("pixel_impedance")

        if "nframes" in kwargs:
            warnings.warn(
                "`nframes` is no longer a constructor argument. Use `len(eitdata)` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _ = kwargs.pop("nframes")

        if kwargs:
            msg = f"Unexpected keyword arguments: {', '.join(kwargs.keys())}."
            raise TypeError(msg)
        return values

    @property
    def pixel_impedance(self) -> np.ndarray:
        """Alias to `values`."""
        return self.values

    @property
    def nframes(self) -> int:
        """Number of frames in the data."""
        warnings.warn("`nframes` is deprecated. Use `len(eitdata)` instead.", DeprecationWarning, stacklevel=2)
        return self.pixel_impedance.shape[0]

    @property
    def framerate(self) -> float:
        """Deprecated alias to `sample_frequency`."""
        warnings.warn(
            "The `framerate` attribute has been deprecated. Use `sample_frequency` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sample_frequency

    @staticmethod
    def ensure_path_list(
        path: str | Path | list[str | Path],
    ) -> list[Path]:
        """Return the path or paths as a list.

        The path of any EITData object can be a single str/Path or a list of str/Path objects. This method returns a
        list of Path objects given either a str/Path or list of str/Paths.
        """
        if isinstance(path, list):
            return [Path(p) for p in path]
        return [Path(path)]

    def __add__(self: Self, other: Self) -> Self:
        return self.concatenate(other)

    def concatenate(self: Self, other: Self, newlabel: str | None = None) -> Self:  # noqa: D102, will be moved to mixin in future
        # Check that data can be concatenated
        self.isequivalent(other, raise_=True)
        if np.min(other.time) <= np.max(self.time):
            msg = f"Concatenation failed. Second dataset ({other.name}) may not start before first ({self.name}) ends."
            raise ValueError(msg)

        self_path = [] if self.path is None else self.ensure_path_list(self.path)
        other_path = [] if other.path is None else self.ensure_path_list(other.path)
        concat_path = [*self_path, *other_path]
        if not concat_path:
            concat_path = None
        newlabel = newlabel or f"Merge of <{self.label}> and <{other.label}>"

        return self.__class__(
            vendor=self.vendor,
            path=concat_path,
            label=self.label,  # TODO: using newlabel leads to errors
            sample_frequency=self.sample_frequency,
            time=np.concatenate((self.time, other.time)),
            pixel_impedance=np.concatenate((self.pixel_impedance, other.pixel_impedance), axis=0),
        )

    def _sliced_copy(
        self,
        start_index: int,
        end_index: int,
        newlabel: str,  # noqa: ARG002
    ) -> Self:
        return self.update(
            time=self.time[start_index:end_index],
            values=self.pixel_impedance[start_index:end_index, :, :],
        )

    def __len__(self):
        return self.pixel_impedance.shape[0]

    def get_summed_impedance(self, *, return_label: str | None = None, **return_kwargs) -> ContinuousData:
        """Return a ContinuousData-object with the same time axis and summed pixel values over time.

        Args:
            return_label: The label of the returned object; defaults to 'summed <label>' where '<label>' is the label of
            the current object.
            **return_kwargs: Keyword arguments for the creation of the returned object.
        """
        summed_impedance = np.nansum(self.pixel_impedance, axis=(1, 2))

        if return_label is None:
            return_label = f"summed {self.label}"

        return_kwargs_: dict[str, Any] = {
            "name": return_label,
            "unit": "AU",
            "category": "impedance",
            "sample_frequency": self.sample_frequency,
        } | return_kwargs

        return ContinuousData(label=return_label, time=self.time, values=summed_impedance, **return_kwargs_)

    def calculate_global_impedance(self) -> np.ndarray:
        """Return the global impedance, i.e. the sum of all included pixels at each frame."""
        return np.nansum(self.pixel_impedance, axis=(1, 2))

    def update(self, **kwargs) -> Self:
        """Return a copy of the object with specified fields replaced.

        Args:
            **kwargs: Fields to replace.

        Returns:
            A new instance of the object with the specified fields replaced.
        """
        if "pixel_impedance" in kwargs:
            if "values" in kwargs:
                msg = "Cannot provide both 'pixel_impedance' and 'values'."
                raise ValueError(msg)
            warnings.warn("`pixel_impedance` has been replaced by `values`.", DeprecationWarning, stacklevel=2)
            kwargs["values"] = kwargs.pop("pixel_impedance")

        if "framerate" in kwargs:
            if "sample_frequency" in kwargs:
                msg = "Cannot provide both 'framerate' and 'sample_frequency'."
                raise ValueError(msg)
            warnings.warn(
                "`framerate` has been deprecated. Use `sample_frequency` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["sample_frequency"] = kwargs.pop("framerate")

        return super().update(**kwargs)


class Vendor(Enum):
    """Enum indicating the vendor (manufacturer) of the source EIT device.

    The enum values are all lowercase strings. For some manufacturers, multiple ways of wrinting are provided mapping to
    the same value, to prevent confusion over conversion of special characters. The "simulated" vendor is provided to
    indicate simulated data.
    """

    DRAEGER = "draeger"
    """Dräger (PulmoVista V500)"""

    TIMPEL = "timpel"
    """Timpel (Enlight 2100)"""

    SENTEC = "sentec"
    """Sentec (Lumon)"""

    DRAGER = DRAEGER
    """Synonym of DRAEGER"""

    DRÄGER = DRAEGER  # noqa: PIE796, PLC2401
    """Synonym of DRAEGER"""

    SIMULATED = "simulated"
    """Simulated data"""
