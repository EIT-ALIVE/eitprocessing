from dataclasses import dataclass
from dataclasses import field
from typing import Any
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self
from typing_extensions import override
from eitprocessing.continuous_data.continuous_data_variant import ContinuousDataVariant
from eitprocessing.mixins.equality import Equivalence
from eitprocessing.variants.variant_collection import VariantCollection


@dataclass(eq=False)
class ContinuousData(Equivalence):
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
        cls.isequivalent(a, b, raise_=True)

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

    @override
    def isequivalent(
        self,
        other: Self,
        raise_: bool = False,
    ) -> bool:
        # fmt: off
        checks = {
            f"Names don't match: {self.name}, {other.name}.": self.name == other.name,
            f"Units don't match: {self.unit}, {other.unit}.": self.unit == other.unit,
            f"Descriptions don't match: {self.description}, {other.description}.": self.description == other.description,
            f"Only one of the datasets is loaded: {self.loaded=}, {other.loaded=}.": self.loaded == other.loaded,
            f"VariantCollections are not equivalent: {self.variants}, {other.variants}.": VariantCollection.isequivalent(self.variants,other.variants, raise_),
        }
        # fmt: on
        return super().isequivalent(other, raise_, checks)


class DataSourceUnknown(Exception):
    """Raised when the source of data is unknown."""
