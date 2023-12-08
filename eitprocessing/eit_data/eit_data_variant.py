"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to when electrical impedance tomographs are read.
"""

import copy
from dataclasses import dataclass
from dataclasses import field
import numpy as np
from typing_extensions import Self
from eitprocessing.mixins.slicing import SelectByIndex
from eitprocessing.variants import Variant


# TODO: make config system
STRICT_EIT_DATA_SHAPE = True


@dataclass
class EITDataVariant(Variant, SelectByIndex):
    pixel_impedance: np.ndarray = field(repr=False, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()

        pi_shape = self.pixel_impedance.shape
        if self.pixel_impedance.ndim != 3 or pi_shape[0] == 0:
            raise ValueError(
                f"Invalid shape {pi_shape} for `pixel_impedance`. Should be (n, 32, 32)."
            )

        if STRICT_EIT_DATA_SHAPE:
            if pi_shape[1:] != (32, 32):
                raise ValueError(
                    f"Invalid shape {pi_shape} for `pixel_impedance`. Should be (n, 32, 32)."
                )

    def __len__(self) -> int:
        return self.pixel_impedance.shape[0]

    def __eq__(self, other: Self) -> bool:
        for attr in ["name", "description", "params"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        if not np.array_equal(
            self.pixel_impedance, other.pixel_impedance, equal_nan=True
        ):
            return False

        return True

    @property
    def global_baseline(self) -> np.ndarray:
        return np.nanmin(self.pixel_impedance)

    @property
    def pixel_impedance_global_offset(self) -> np.ndarray:
        return self.pixel_impedance - self.global_baseline

    @property
    def pixel_baseline(self) -> np.ndarray:
        return np.nanmin(self.pixel_impedance, axis=0)

    @property
    def pixel_impedance_individual_offset(self) -> np.ndarray:
        return self.pixel_impedance - np.min(self.pixel_impedance, axis=0)

    @property
    def global_impedance(self) -> np.ndarray:
        return np.nansum(self.pixel_impedance, axis=(1, 2))

    def concatenate(self, other: Self) -> Self:
        self.check_equivalence(other, raise_=True)

        return self.__class__(
            name=self.name,
            description=self.description,
            params=self.params,
            pixel_impedance=np.concatenate(
                [self.pixel_impedance, other.pixel_impedance], axis=0
            ),
        )

    def _sliced_copy(
        self, start_index: int, end_index: int | None = None, label: str | None = None
    ) -> Self:
        label = label or f"Slice ({start_index}-{end_index}) of <{self.name}>"
        pixel_impedance = self.pixel_impedance[start_index:end_index, :, :]

        return self.__class__(
            name=label,
            description=self.description,
            params=copy.deepcopy(self.params),
            pixel_impedance=pixel_impedance,
        )

    def copy(self, label: str | None = None) -> Self:
        label = label or f"Copy of <{self.name}>"
        return self.__class__(
            name=label,
            description=self.description,
            params=copy.deepcopy(self.params),
            pixel_impedance=np.copy(self.pixel_impedance),
        )
