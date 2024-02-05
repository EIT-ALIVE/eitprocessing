from dataclasses import dataclass
from dataclasses import field
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self
from eitprocessing.mixins.slicing import SelectByTime
from eitprocessing.variants import Variant


@dataclass(eq=False)
class EITDataVariant(Variant, SelectByTime):
    _data_field_name: str = "pixel_impedance"
    pixel_impedance: NDArray = field(repr=False, kw_only=True)

    def __len__(self):
        return self.pixel_impedance.shape[0]

    @property
    def global_baseline(self):
        return np.nanmin(self.pixel_impedance)

    @property
    def pixel_impedance_global_offset(self):
        return self.pixel_impedance - self.global_baseline

    @property
    def pixel_baseline(self):
        return np.nanmin(self.pixel_impedance, axis=0)

    @property
    def pixel_impedance_individual_offset(self):
        return self.pixel_impedance - np.min(self.pixel_impedance, axis=0)

    @property
    def global_impedance(self):
        return np.nansum(self.pixel_impedance, axis=(1, 2))

    @classmethod
    def concatenate(cls, a: Self, b: Self) -> Self:
        cls.isequivalent(a, b, raise_=True)

        return cls(
            label=a.label,
            description=a.description,
            params=a.params,
            pixel_impedance=np.concatenate(
                [a.pixel_impedance, b.pixel_impedance], axis=0
            ),
        )

    def _sliced_copy(
        self,
        start_index: int,
        end_index: int,
        label: str | None = None,
    ) -> Self:
        pixel_impedance = self.pixel_impedance[start_index:end_index, :, :]

        return self.__class__(
            label=label,
            description=self.description,
            params=self.params,
            pixel_impedance=pixel_impedance,
        )

    def copy(self, label: str | None = None):
        label = label or f"Copy of <{self.label}>"
        return self.__class__(
            label=label,
            description=self.description,
            params=self.params,
            pixel_impedance=np.copy(self.pixel_impedance),
        )
