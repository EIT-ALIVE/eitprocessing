from dataclasses import dataclass, field

from numpy.typing import NDArray
from typing_extensions import Self

from eitprocessing.variants import Variant


@dataclass(eq=False)
class ContinuousDataVariant(Variant):
    values: NDArray = field(kw_only=True)

    def __len__(self) -> int:
        return len(self.values)

    @classmethod
    def concatenate(cls, a: Self, b: Self) -> Self:
        return super().concatenate(a, b)
