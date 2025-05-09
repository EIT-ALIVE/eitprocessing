from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from strenum import LowercaseStrEnum

from eitprocessing.datahandling import DataContainer
from eitprocessing.datahandling.mixins.slicing import SelectByTime

if TYPE_CHECKING:
    from typing_extensions import Self


T = TypeVar("T", bound="EITData")


@dataclass(eq=False)
class EITData(DataContainer, SelectByTime):
    """Container for EIT impedance data.

    This class holds the pixel impedance from an EIT measurement, as well as metadata describing the measurement. The
    class is meant to hold data from (part of) a singular continuous measurement.

    This class can't be initialized directly. Instead, use `load_eit_data(<path>, vendor=<vendor>)` to load data from
    disk.

    Args:
        path: The path of list of paths of the source from which data was derived.
        nframes: Number of frames.
        time: The time of each frame (since start measurement).
        sample_frequency: The (average) frequency at which the frames are collected, in Hz.
        vendor: The vendor of the device the data was collected with.
        label: Computer readable label identifying this dataset.
        name: Human readable name for the data.
        pixel_impedance: Impedance values for each pixel at each frame.
    """  # TODO: fix docstring

    path: str | Path | list[Path | str] = field(compare=False, repr=False)
    nframes: int = field(repr=False)
    time: np.ndarray = field(repr=False)
    sample_frequency: float = field(metadata={"check_equivalence": True}, repr=False)
    vendor: Vendor = field(metadata={"check_equivalence": True}, repr=False)
    label: str | None = field(default=None, compare=False, metadata={"check_equivalence": True})
    description: str = field(default="", compare=False, repr=False)
    name: str | None = field(default=None, compare=False, repr=False)
    pixel_impedance: np.ndarray = field(repr=False, kw_only=True)

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.__class__.__name__}_{id(self)}"

        self.path = self.ensure_path_list(self.path)
        if len(self.path) == 1:
            self.path = self.path[0]

        self.name = self.name or self.label

        if (lv := len(self.pixel_impedance)) != (lt := len(self.time)):
            msg = f"The number of time points ({lt}) does not match the number of pixel impedance values ({lv})."
            raise ValueError(msg)

    @property
    def framerate(self) -> float:
        """Deprecated alias to `sample_frequency`."""
        warnings.warn(
            "The `framerate` attribute has been deprecated. Use `sample_frequency` instead.",
            DeprecationWarning,
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

        self_path = self.ensure_path_list(self.path)
        other_path = self.ensure_path_list(other.path)
        newlabel = newlabel or f"Merge of <{self.label}> and <{other.label}>"

        return self.__class__(
            vendor=self.vendor,
            path=[*self_path, *other_path],
            label=self.label,  # TODO: using newlabel leads to errors
            sample_frequency=self.sample_frequency,
            nframes=self.nframes + other.nframes,
            time=np.concatenate((self.time, other.time)),
            pixel_impedance=np.concatenate((self.pixel_impedance, other.pixel_impedance), axis=0),
        )

    def _sliced_copy(
        self,
        start_index: int,
        end_index: int,
        newlabel: str,  # noqa: ARG002
    ) -> Self:
        cls = self.__class__
        time = np.copy(self.time[start_index:end_index])
        nframes = len(time)

        pixel_impedance = np.copy(self.pixel_impedance[start_index:end_index, :, :])

        return cls(
            path=self.path,
            nframes=nframes,
            vendor=self.vendor,
            time=time,
            sample_frequency=self.sample_frequency,
            label=self.label,  # newlabel gives errors
            pixel_impedance=pixel_impedance,
        )

    def __len__(self):
        return self.pixel_impedance.shape[0]

    def calculate_global_impedance(self) -> np.ndarray:
        """Return the global impedance, i.e. the sum of all included pixels at each frame."""
        return np.nansum(self.pixel_impedance, axis=(1, 2))


class Vendor(LowercaseStrEnum):
    """Enum indicating the vendor (manufacturer) of the source EIT device."""

    DRAEGER = auto()
    TIMPEL = auto()
    SENTEC = auto()
    DRAGER = DRAEGER
    DRÄGER = DRAEGER  # noqa: PLC2401
