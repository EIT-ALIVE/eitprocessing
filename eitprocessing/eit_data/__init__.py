from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from typing_extensions import Self, override

from eitprocessing.eit_data.vendor import Vendor
from eitprocessing.mixins.equality import Equivalence
from eitprocessing.mixins.slicing import SelectByTime

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from eitprocessing.data_collection import DataCollection

T = TypeVar("T", bound="EITData")


@dataclass(eq=False)
class EITData(SelectByTime, Equivalence, ABC):
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
    time: NDArray
    framerate: float
    vendor: Vendor
    phases: list = field(default_factory=list)
    events: list = field(default_factory=list)
    label: str | None = None
    pixel_impedance: NDArray = field(repr=False, kw_only=True)

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.__class__.__name__}_{id(self)}"
        self._check_equivalence = ["vendor", "framerate"]

    @classmethod
    def from_path(
        cls,
        path: str | Path | list[str | Path],
        vendor: Vendor | str,
        framerate: float | None = None,
        first_frame: int = 0,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> DataCollection | tuple[DataCollection, DataCollection, DataCollection]:
        """Load sequence from path(s).

        Args:
            path: relative or absolute path(s) to data file.
            vendor: vendor indicating the device used.
            label: description of object for human interpretation.
                Defaults to "Sequence_<unique_id>".
            framerate: framerate at which the data was recorded.
                Default for Draeger: 20
                Default for Timpel: 50
                Default for Sentec: 50.2
            first_frame: index of first frame to load.
                Defaults to 0.
            max_frames: maximum number of frames to load.
                The actual number of frames can be lower than this if this
                would surpass the final frame.
            return_non_eit_data: whether to load available continuous and sparse data.

        Raises:
            NotImplementedError: is raised when there is no loading method for
            the given vendor.

        Returns:
            EITData: container for the loaded data and metadata from all files in path.
        """
        from eitprocessing.data_collection import DataCollection

        vendor = cls._ensure_vendor(vendor)
        vendor_class = cls._get_vendor_class(vendor)

        first_frame = cls._check_first_frame(first_frame)

        paths = cls._ensure_path_list(path)

        eit_datasets: list[DataCollection] = []
        continuous_datasets: list[DataCollection] = []
        sparse_datasets: list[DataCollection] = []

        for single_path in paths:
            single_path.resolve(strict=True)  # raise error if any file does not exist

        for single_path in paths:
            loaded_data = vendor_class._from_path(  # noqa: SLF001
                path=single_path,
                framerate=framerate,
                first_frame=first_frame,
                max_frames=max_frames,
                return_non_eit_data=return_non_eit_data,
            )

            if return_non_eit_data:
                from eitprocessing.continuous_data import ContinuousData
                from eitprocessing.sparse_data import SparseData

                if type(loaded_data) is not tuple:
                    eit = loaded_data
                    continuous = DataCollection(ContinuousData)
                    sparse = DataCollection(SparseData)
                else:
                    eit, continuous, sparse = loaded_data

                eit: DataCollection
                continuous: DataCollection
                sparse: DataCollection

                eit_datasets.append(eit)
                continuous_datasets.append(continuous)
                sparse_datasets.append(sparse)

            else:
                assert isinstance(loaded_data, DataCollection)  # noqa: S101
                eit_datasets.append(loaded_data)

        if return_non_eit_data:
            return tuple(
                reduce(DataCollection.concatenate, datasets)
                for datasets in (eit_datasets, continuous_datasets, sparse_datasets)
            )

        return reduce(DataCollection.concatenate, eit_datasets)

    @classmethod
    @abstractmethod
    def _from_path(
        cls: type[Self],
        path: Path,
        framerate: float | None = None,
        first_frame: int | None = None,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> DataCollection | tuple[DataCollection, DataCollection, DataCollection]: ...

    @staticmethod
    def _ensure_path_list(path: str | Path | list[str | Path]) -> list[Path]:
        if isinstance(path, list):
            return [Path(p) for p in path]
        return [Path(path)]

    @staticmethod
    def _get_vendor_class(vendor: Vendor) -> type[EITData_]:
        from eitprocessing.eit_data.draeger import DraegerEITData
        from eitprocessing.eit_data.sentec import SentecEITData
        from eitprocessing.eit_data.timpel import TimpelEITData

        vendor_classes: dict[Vendor, type[EITData_]] = {
            Vendor.DRAEGER: DraegerEITData,
            Vendor.TIMPEL: TimpelEITData,
            Vendor.SENTEC: SentecEITData,
        }
        return vendor_classes[vendor]

    @staticmethod
    def _check_first_frame(first_frame: int | None) -> int:
        if first_frame is None:
            first_frame = 0
        if int(first_frame) != first_frame:
            msg = f"`first_frame` must be an int, but was given as {first_frame} (type: {type(first_frame)})"
            raise TypeError(msg)
        if first_frame < 0:
            msg = f"`first_frame` can not be negative, but was given as {first_frame}"
            raise ValueError(msg)
        return int(first_frame)

    @staticmethod
    def _ensure_vendor(vendor: Vendor | str) -> Vendor:
        """Check whether vendor exists, and assure it's a Vendor object."""
        try:
            return Vendor(vendor)
        except ValueError as e:
            msg = f"Unknown vendor {vendor}."
            raise UnknownVendorError(msg) from e

    def concatenate(self: T, other: T, label: str | None = None) -> T:  # noqa: D102, will be removed soon
        cls = self.__class__
        self.isequivalent(other, raise_=True)

        a_path = cls._ensure_path_list(self.path)
        b_path = cls._ensure_path_list(other.path)
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

        cls_ = cls._get_vendor_class(self.vendor)

        phases = self.phases + other.phases
        events = self.events + other.events

        return cls_(
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
        cls = self._get_vendor_class(self.vendor)

        if end_index <= start_index:
            msg = f"{end_index} (end_index) lower than {start_index} (start_index)."
            raise ValueError(
                msg,
            )

        time_length = len(self.time)

        if start_index >= time_length:
            raise ValueError(
                f"{start_index} (start_index) higher than maximum time length {time_length}."
            )

        if end_index >= time_length:
            raise ValueError(
                f"{end_index} (end_index) higher than maximum time length {time_length}."
            )

        time = self.time[start_index:end_index]
        nframes = len(time)

        phases = list(filter(lambda p: start_index <= p.index < end_index, self.phases))
        events = list(filter(lambda e: start_index <= e.index < end_index, self.events))

        pixel_impedance = self.pixel_impedance[start_index:end_index, :, :]

        return cls(
            path=self.path,
            nframes=nframes,
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

    @property
    def global_impedance(self) -> np.ndarray:
        """Return the global impedance, i.e. the sum of all pixels at each frame."""
        return np.nansum(self.pixel_impedance, axis=(1, 2))


@dataclass(eq=False)
class EITData_(EITData):  # noqa: N801, D101
    vendor: Vendor = field(init=False)

    def __add__(self: T, other: T) -> T:
        return self.concatenate(self, other)

    @override  # remove vendor as argument
    @classmethod
    def from_path(
        cls,
        path: str | Path | list[str | Path],
        framerate: float | None = None,
        first_frame: int = 0,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> DataCollection | tuple[DataCollection, DataCollection, DataCollection]:
        return super().from_path(
            path=path,
            vendor=cls.vendor,
            framerate=framerate,
            first_frame=first_frame,
            max_frames=max_frames,
            return_non_eit_data=return_non_eit_data,
        )


class UnknownVendorError(ValueError):
    """Raised when an unknown vendor is provided when trying to load data."""
