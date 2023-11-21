import contextlib
from dataclasses import dataclass
from dataclasses import field
from typing import Any
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self
from ..helper import NotEquivalent
from ..variants.variant_collection import VariantCollection
from .continuous_data_variant import ContinuousDataVariant


@dataclass
class ContinuousData:
    name: str
    unit: str
    description: str
    time: NDArray
    loaded: bool
    calculated_from: Any | list[Any] | None = None
    variants: VariantCollection = field(
        default_factory=lambda: VariantCollection(ContinuousDataVariant)
    )

    def __post_init__(self):
        if not self.loaded and not self.calculated_from:
            raise DataSourceUnknown(
                "Data must be loaded or calculated form another dataset."
            )

    @classmethod
    def concatenate(cls, a: Self, b: Self) -> Self:
        cls.check_equivalence(a, b, raise_=True)

        calculcated_from = None if a.loaded else [a.calculated_from, b.calculated_from]

        time = np.concatenate([a.time, b.time])

        obj = cls(
            name=a.name,
            unit=a.unit,
            description=a.description,
            time=time,
            loaded=a.loaded,
            calculated_from=calculcated_from,
            variants=VariantCollection.concatenate(a.variants, b.variants),
        )
        return obj

    @classmethod
    def check_equivalence(cls, a: Self, b: Self, raise_: bool = False) -> bool:
        cm = contextlib.nullcontext() if raise_ else contextlib.suppress(NotEquivalent)
        with cm:
            if a.name != b.name:
                raise NotEquivalent(f"Names do not match: {a.name}, {b.name}")
            if a.unit != b.unit:
                raise NotEquivalent(f"Units do not match: {a.unit}, {b.unit}")
            if a.description != b.description:
                raise NotEquivalent(
                    f"Descriptions do not match: {a.description}, {b.description}"
                )
            if a.loaded != b.loaded:
                raise NotEquivalent(
                    f"Only one of the datasets is loaded: {a.loaded=}, {b.loaded=}"
                )

            VariantCollection.check_equivalence(a.variants, b.variants, raise_=True)

            return True

        return False


class DataSourceUnknown(Exception):
    """Raised when the source of data is unknown."""
