from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from eitprocessing.continuous_data.continuous_data_variant import ContinuousDataVariant
from eitprocessing.mixins.equality import Equivalence
from eitprocessing.variants.variant_collection import VariantCollection


@dataclass(eq=False)
class ContinuousData(Equivalence):
    name: str
    unit: str
    description: str
    category: str
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
        self._check_equivalence = ["unit", "category"]

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


class DataSourceUnknown(Exception):
    """Raised when the source of data is unknown."""
