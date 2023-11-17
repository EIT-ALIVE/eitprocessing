import functools
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TypeAlias
from typing import TypeVar
from typing import Union
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self
from eitprocessing.variants.variant_collection import VariantCollection
from ..helper import NotEquivalent
from .vendor import Vendor


T = TypeVar("T", bound="EITData")
PathLike: TypeAlias = Union[str, Path]


@dataclass
class EITData(ABC):
    path: Path | list[Path]
    nframes: int
    time: NDArray
    framerate: float
    variants: VariantCollection
    vendor: Vendor
    phases: list = field(default_factory=list)
    events: list = field(default_factory=list)
    label: str | None = None

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.__class__.__name__}_{id(self)}"

    @classmethod
    def from_path(  # pylint: disable=too-many-arguments
        cls,
        path: PathLike | list[PathLike],
        vendor: Vendor | str,
        label: str | None = None,
        framerate: float | None = None,
        first_frame: int = 0,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> Self:
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

        sequences = []

        for single_path in paths:
            single_path.resolve(strict=True)  # raises if file does not exists
            sequences.append(
                vendor_class._from_path(  # pylint: disable=protected-access
                    path=single_path,
                    label=label,
                    framerate=framerate,
                    first_frame=first_frame,
                    max_frames=max_frames,
                    return_non_eit_data=return_non_eit_data,
                )
            )
        return functools.reduce(cls.concatenate, sequences)

    @staticmethod
    def _ensure_path_list(path: PathLike | list[PathLike]) -> list[Path]:
        if isinstance(path, list):
            return [Path(p) for p in path]
        return [Path(path)]

    @staticmethod
    def _get_vendor_class(vendor: Vendor):
        from .draeger import DraegerEITData  # pylint: disable=import-outside-toplevel
        from .sentec import SentecEITData  # pylint: disable=import-outside-toplevel
        from .timpel import TimpelEITData  # pylint: disable=import-outside-toplevel

        vendor_classes = {
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
    @abstractmethod
    def _from_path(  # pylint: disable=too-many-arguments
        cls,
        path: Path,
        label: str | None,
        framerate: float | None,
        first_frame: int | None,
        max_frames: int | None,
    ):
        ...

    def __add__(self, other):
        return self.__class__.concatenate(self, other)

    @classmethod
    def concatenate(cls, a: Self, b: Self, label: str | None = None) -> Self:
        cls.check_equivalence(a, b, raise_=True)

        subclass = cls._get_vendor_class(a.vendor)

        a_path = cls._ensure_path_list(a.path)
        b_path = cls._ensure_path_list(b.path)
        path = a_path + b_path

        label = label or f"Concatenation of <{a.label}> and <{b.label}>"
        framerate = a.framerate
        nframes = a.nframes + b.nframes
        time = np.concatenate((a.time, b.time))
        variants = VariantCollection.concatenate(a.variants, b.variants)

        return subclass(
            path=path,
            label=label,
            framerate=framerate,
            nframes=nframes,
            time=time,
            variants=variants,
        )

    @classmethod
    def check_equivalence(cls, a: T, b: T, raise_=False) -> bool:
        try:
            if a.__class__ != b.__class__:
                raise NotEquivalent(f"Classes don't match: {type(a)}, {type(b)}")

            if a.framerate != b.framerate:
                raise NotEquivalent(
                    f"Framerates do not match: {a.framerate}, {b.framerate}"
                )

            VariantCollection.check_equivalence(a.variants, b.variants, raise_=True)

        except NotEquivalent:
            # re-raises the exceptions if raise_ is True, or returns False
            if raise_:
                raise
            return False

        return True


class NoVendorProvided(Exception):
    """Raised when no vendor is provided when trying to load data."""


class UnknownVendor(Exception):
    """Raised when an unknown vendor is provided when trying to load data."""
