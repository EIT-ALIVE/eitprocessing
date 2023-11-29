from __future__ import annotations
import contextlib
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from functools import reduce
from pathlib import Path
from typing import TypeAlias
from typing import TypeVar
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self
from typing_extensions import override
from eitprocessing.continuous_data.continuous_data_collection import (
    ContinuousDataCollection,
)
from eitprocessing.eit_data.eit_data_variant import EITDataVariant
from eitprocessing.mixins.slicing import SelectByTime
from eitprocessing.sparse_data.sparse_data_collection import SparseDataCollection
from eitprocessing.variants.variant_collection import VariantCollection
from ..helper import NotEquivalent
from .vendor import Vendor


PathLike: TypeAlias = str | Path
PathArg: TypeAlias = PathLike | list[PathLike]
T = TypeVar("T", bound="EITData")


@dataclass
class EITData(SelectByTime, ABC):
    path: Path | list[Path]
    nframes: int
    time: NDArray
    framerate: float
    vendor: Vendor
    phases: list = field(default_factory=list)
    events: list = field(default_factory=list)
    label: str | None = None
    variants: VariantCollection = field(
        default_factory=lambda: VariantCollection(EITDataVariant)
    )

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.__class__.__name__}_{id(self)}"

    @classmethod
    def from_path(  # pylint: disable=too-many-arguments,too-many-locals
        cls,
        path: PathArg,
        vendor: Vendor | str,
        label: str | None = None,
        framerate: float | None = None,
        first_frame: int = 0,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> Self | tuple[Self, ContinuousDataCollection, SparseDataCollection]:
        """Load sequence from path(s)

        Args:
            path (Path | str | list[Path | str]): path(s) to data file.
            vendor (Vendor | str): vendor indicating the device used.
            label (str): description of object for human interpretation.
                Defaults to "Sequence_<unique_id>".
            framerate (int, optional): framerate at which the data was recorded.
                Default for Draeger: 20
                Default for Timpel: 50
            first_frame (int, optional): index of first time point of sequence
                (i.e. NOT the timestamp).
                Defaults to 0.
            max_frames (int, optional): maximum number of frames to load.
                The actual number of frames can be lower than this if this
                would surpass the final frame.

        Raises:
            NotImplementedError: is raised when there is no loading method for
            the given vendor.

        Returns:
            Sequence: a sequence containing the loaded data from all files in path
        """

        vendor = cls._ensure_vendor(vendor)
        vendor_class = cls._get_vendor_class(vendor)

        first_frame = cls._check_first_frame(first_frame)

        paths = cls._ensure_path_list(path)

        eit_datasets: list[EITData] = []
        continuous_datasets: list[ContinuousDataCollection] = []
        sparse_datasets: list[SparseDataCollection] = []

        for single_path in paths:
            # this checks whether each path exists before any path is loaded to
            # prevent unneccesary loading
            single_path.resolve(strict=True)  # raises if file does not exists

        for single_path in paths:
            loaded_data = vendor_class._from_path(  # pylint: disable=protected-access
                path=single_path,
                label=label,
                framerate=framerate,
                first_frame=first_frame,
                max_frames=max_frames,
                return_non_eit_data=return_non_eit_data,
            )

            if return_non_eit_data:
                eit, continuous, sparse = loaded_data

                # assertions for type checking
                assert isinstance(eit, EITData)
                assert isinstance(continuous, ContinuousDataCollection)
                assert isinstance(sparse, SparseDataCollection)

                eit_datasets.append(eit)
                continuous_datasets.append(continuous)
                sparse_datasets.append(sparse)

            else:
                assert isinstance(loaded_data, EITData)
                eit_datasets.append(loaded_data)

        if return_non_eit_data:
            return (
                reduce(cls.concatenate, eit_datasets),
                reduce(ContinuousDataCollection.concatenate, continuous_datasets),
                reduce(SparseDataCollection.concatenate, sparse_datasets),
            )

        return reduce(cls.concatenate, eit_datasets)

    @staticmethod
    def _ensure_path_list(path: PathArg) -> list[Path]:
        if isinstance(path, list):
            return [Path(p) for p in path]
        return [Path(path)]

    @staticmethod
    def _get_vendor_class(vendor: Vendor) -> type[EITData_]:
        from .draeger import DraegerEITData  # pylint: disable=import-outside-toplevel
        from .sentec import SentecEITData  # pylint: disable=import-outside-toplevel
        from .timpel import TimpelEITData  # pylint: disable=import-outside-toplevel

        vendor_classes: dict[Vendor, type[EITData_]] = {
            Vendor.DRAEGER: DraegerEITData,
            Vendor.TIMPEL: TimpelEITData,
            Vendor.SENTEC: SentecEITData,
        }
        subclass = vendor_classes[vendor]
        return subclass

    @staticmethod
    def _check_first_frame(first_frame):
        if first_frame is None:
            first_frame = 0
        if int(first_frame) != first_frame:
            raise TypeError(
                f"`first_frame` must be an int, but was given as"
                f" {first_frame} (type: {type(first_frame)})"
            )
        if first_frame < 0:
            raise ValueError(
                f"`first_frame` can not be negative, but was given as {first_frame}"
            )
        first_frame = int(first_frame)
        return first_frame

    @staticmethod
    def _ensure_vendor(vendor: Vendor | str) -> Vendor:
        """Check whether vendor exists, and assure it's a Vendor object."""
        if not vendor:
            raise NoVendorProvided()

        try:
            return Vendor(vendor)
        except ValueError as e:
            raise UnknownVendor(f"Unknown vendor {vendor}.") from e

    @classmethod
    def concatenate(cls, a: T, b: T, label: str | None = None) -> T:
        cls.check_equivalence(a, b, raise_=True)

        a_path = cls._ensure_path_list(a.path)
        b_path = cls._ensure_path_list(b.path)
        path = a_path + b_path

        if np.min(b.time) <= np.max(a.time):
            raise ValueError(f"{b} (b) starts before {a} (a) ends.")
        time = np.concatenate((a.time, b.time))

        label = label or f"Concatenation of <{a.label}> and <{b.label}>"
        framerate = a.framerate
        nframes = a.nframes + b.nframes
        variants = VariantCollection.concatenate(a.variants, b.variants)

        cls_ = cls._get_vendor_class(a.vendor)

        return cls_(
            path=path,
            label=label,
            framerate=framerate,
            nframes=nframes,
            time=time,
            variants=variants,
        )

    @classmethod
    def check_equivalence(cls, a: T, b: T, raise_=False) -> bool:
        cm = contextlib.nullcontext() if raise_ else contextlib.suppress(NotEquivalent)
        with cm:
            if a.__class__ != b.__class__:
                raise NotEquivalent(f"Classes don't match: {type(a)}, {type(b)}")

            if a.framerate != b.framerate:
                raise NotEquivalent(
                    f"Framerates do not match: {a.framerate}, {b.framerate}"
                )

            VariantCollection.check_equivalence(a.variants, b.variants, raise_=True)

            return True

        return False

    def _sliced_copy(
        self: Self, start_index: int, end_index: int, label: str | None = None
    ) -> Self:
        cls = self._get_vendor_class(self.vendor)
        time = self.time[start_index:end_index]
        nframes = len(time)

        phases = list(filter(lambda p: start_index <= p.index < end_index, self.phases))
        events = list(filter(lambda e: start_index <= e.index < end_index, self.events))

        obj = cls(
            path=self.path,
            nframes=nframes,
            time=time,
            framerate=self.framerate,
            phases=phases,
            events=events,
            label=label,
        )

        for variant in self.variants.values():
            obj.variants.add(
                variant._sliced_copy(  # pylint: disable=protected-access
                    start_index, end_index
                )
            )

        return obj

    @classmethod
    @abstractmethod
    def _from_path(  # pylint: disable=too-many-arguments
        cls: type[Self],
        path: Path,
        label: str | None = None,
        framerate: float | None = None,
        first_frame: int | None = None,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> Self | tuple[Self, ContinuousDataCollection, SparseDataCollection]:
        ...


@dataclass
class EITData_(EITData):
    vendor: Vendor = field(init=False)

    def __add__(self: T, other: T) -> T:
        return self.concatenate(self, other)

    @override  # remove vendor as argument
    @classmethod
    def from_path(  # pylint: disable=too-many-arguments,arguments-differ
        cls,
        path: PathArg,
        label: str | None = None,
        framerate: float | None = None,
        first_frame: int = 0,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> Self | tuple[Self, ContinuousDataCollection, SparseDataCollection]:
        return super().from_path(
            path,
            cls.vendor,
            label,
            framerate,
            first_frame,
            max_frames,
            return_non_eit_data,
        )


class NoVendorProvided(Exception):
    """Raised when no vendor is provided when trying to load data."""


class UnknownVendor(Exception):
    """Raised when an unknown vendor is provided when trying to load data."""
