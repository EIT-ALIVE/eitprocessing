from __future__ import annotations

from dataclasses import dataclass, field
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from strenum import LowercaseStrEnum
from typing_extensions import Self

from eitprocessing.datahandling.mixins.equality import Equivalence
from eitprocessing.datahandling.mixins.slicing import SelectByTime

if TYPE_CHECKING:
    from eitprocessing.datahandling.eitdata import Vendor

T = TypeVar("T", bound="EITData")


@dataclass(eq=False)
class EITData(SelectByTime, Equivalence):
    """Container for EIT data.

    This class holds the pixel impedance from an EIT measurement, as well as metadata describing the measurement. The
    class is meant to hold data from (part of) a singular continuous measurement.

    This class can't be initialized directly. Instead, use `EITData.from_path(...)` to load data from disk.
    Currently, loading data from three vendors is supported. You can either pass the vendor when using
    `EITData.from_path(..., vendor="timpel")`, or use one of the available subclasses of EITData:
    `SentecEITData.from_path(...)`.

    Several convenience methods are supplied for calculating global impedance, calculating or removing baselines, etc.
    """

    path: Path | list[Path]
    nframes: int
    time: np.ndarray = field(repr=False)
    framerate: float
    vendor: Vendor
    phases: list = field(default_factory=list, repr=False)
    events: list = field(default_factory=list, repr=False)
    label: str | None = None
    pixel_impedance: np.ndarray = field(repr=False, kw_only=True)

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.__class__.__name__}_{id(self)}"
        self._check_equivalence = ["vendor", "framerate"]

    @staticmethod
    def ensure_path_list(path: str | Path | list[str | Path]) -> list[Path]:
        """Return the path or paths as a list.

        The path of any EITData object can be a single str/Path or a list of str/Path objects. This method returns a
        list of Path objects given either a str/Path or list of str/Paths.
        """
        if isinstance(path, list):
            return [Path(p) for p in path]
        return [Path(path)]

    def __add__(self: T, other: T) -> T:
        return self.concatenate(self, other)

    def concatenate(self: T, other: T, label: str | None = None) -> T:  # noqa: D102, will be removed soon
        cls = self.__class__
        self.isequivalent(other, raise_=True)

        a_path = cls.ensure_path_list(self.path)
        b_path = cls.ensure_path_list(other.path)
        path = a_path + b_path

        if np.min(other.time) <= np.max(self.time):
            msg = f"{other} (b) starts before {self} (a) ends."
            raise ValueError(msg)
        time = np.concatenate((self.time, other.time))

        pixel_impedance = np.concatenate((self.pixel_impedance, other.pixel_impedance), axis=0)

        if self.label != other.label:
            msg = "Can't concatenate data with different labels."
            raise ValueError(msg)

        label = self.label
        framerate = self.framerate
        nframes = self.nframes + other.nframes

        phases = self.phases + other.phases
        events = self.events + other.events

        return self.__class__(
            vendor=self.vendor,
            path=path,
            label=label,
            framerate=framerate,
            nframes=nframes,
            time=time,
            pixel_impedance=pixel_impedance,
            phases=phases,
            events=events,
        )

    def _sliced_copy(
        self,
        start_index: int,
        end_index: int,
        label: str,
    ) -> Self:
        cls = self.__class__
        time = self.time[start_index:end_index]
        nframes = len(time)

        phases = list(filter(lambda p: start_index <= p.index < end_index, self.phases))
        events = list(filter(lambda e: start_index <= e.index < end_index, self.events))

        pixel_impedance = self.pixel_impedance[start_index:end_index, :, :]

        return cls(
            path=self.path,
            nframes=nframes,
            vendor=self.vendor,
            time=time,
            framerate=self.framerate,
            phases=phases,
            events=events,
            label=label,
            pixel_impedance=pixel_impedance,
        )

    def __len__(self):
        return self.pixel_impedance.shape[0]

    @property
    def global_baseline(self) -> np.ndarray:
        """Return the global baseline, i.e. the minimum pixel impedance across all pixels."""
        return np.nanmin(self.pixel_impedance)

    @property
    def pixel_impedance_global_offset(self) -> np.ndarray:
        """Return the pixel impedance with the global baseline removed.

        In the resulting array the minimum impedance across all pixels is set to 0.
        """
        return self.pixel_impedance - self.global_baseline

    @property
    def pixel_baseline(self) -> np.ndarray:
        """Return the lowest value in each individual pixel over time."""
        return np.nanmin(self.pixel_impedance, axis=0)

    @property
    def pixel_impedance_individual_offset(self) -> np.ndarray:
        """Return the pixel impedance with the baseline of each individual pixel removed.

        Each pixel in the resulting array has a minimum value of 0.
        """
        return self.pixel_impedance - self.pixel_baseline

    def _calculate_global_impedance(self) -> np.ndarray:
        """Return the global impedance, i.e. the sum of all pixels at each frame."""
        return np.nansum(self.pixel_impedance, axis=(1, 2))


class Vendor(LowercaseStrEnum):
    """Enum indicating the vendor (manufacturer) of the source EIT device."""

    DRAEGER = auto()
    TIMPEL = auto()
    SENTEC = auto()
    DRAGER = DRAEGER
    DRÄGER = DRAEGER
