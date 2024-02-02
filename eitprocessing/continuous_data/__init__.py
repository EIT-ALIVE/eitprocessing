from dataclasses import dataclass
from dataclasses import field
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
    loaded: bool
    calculated_from: Any | list[Any] | None = None
    variants: VariantCollection = field(
        default_factory=lambda: VariantCollection(ContinuousDataVariant)
    )

    def __post_init__(self):
        if not self.loaded and not self.calculated_from:
            raise ValueError("Data must be loaded or calculated form another dataset.")
