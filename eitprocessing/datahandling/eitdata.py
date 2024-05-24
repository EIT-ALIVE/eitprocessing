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

    This class can't be initialized directly. Instead, use `load_eit_data(<path>, vendor=<vendor>)` to load data from
    disk.

    Args:
        path: the path of list of paths of the source from which data was derived.
        nframes: number of frames
        time:

    Several convenience methods are supplied for calculating global impedance, calculating or removing baselines, etc.
    """  # TODO: fix docstring

    path: Path | list[Path] = field(compare=False)
    nframes: int
    time: np.ndarray = field(repr=False)
    framerate: float = field(metadata={"check_equivalence": True})
    vendor: Vendor = field(metadata={"check_equivalence": True})
    label: str | None = field(default=None, compare=False, metadata={"check_equivalence": True})
    name: str | None = field(default=None, compare=False)
    pixel_impedance: np.ndarray = field(repr=False, kw_only=True)

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.__class__.__name__}_{id(self)}"
        self.name = self.name or self.label

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
        return self.concatenate(other)

    def concatenate(self: T, other: T, newlabel: str | None = None) -> T:  # noqa: D102, will be moved to mixin in future
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
            path=self_path + other_path,
            label=self.label,  # TODO: using newlabel leads to errors
            framerate=self.framerate,
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
        time = self.time[start_index:end_index]
        nframes = len(time)

        pixel_impedance = self.pixel_impedance[start_index:end_index, :, :]

        return cls(
            path=self.path,
            nframes=nframes,
            vendor=self.vendor,
            time=time,
            framerate=self.framerate,
            label=self.label,  # newlabel gives errors
            pixel_impedance=pixel_impedance,
        )

    def __len__(self):
        return self.pixel_impedance.shape[0]

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
