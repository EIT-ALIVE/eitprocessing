"""Region of Interest selection and data masking.

This module contains tools for the selection of regions of interest and masking data. The central class of this module
is `Mask`. Any type of region of interest selection results in a `Mask` object. A mask can be applied to any dataset.

Several default masks have been predefined:

- `VENTRAL_MASK` includes only the first 16 rows;
- `DORSAL_MASK` includes only the last 16 rows;
- `RIGHT_MASK` includes only the first 16 columns (NB: right means right lung, which is te left side of the EIT image);
- `LEFT_MASK` includes only the last 16 columns (NB: left means left lung, which is te right side of the EIT image);
- `QUADRANT_1_MASK` includes the top right quadrant (NB: right means right lung, which is te left side of the EIT
  image);
- `QUADRANT_2_MASK` includes the top left quadrant (NB: left means left lung, which is te right side of the EIT image);
- `QUADRANT_3_MASK` includes the bottom right quadrant (NB: right means right lung, which is te left side of the EIT
  image);
- `QUADRANT_4_MASK` includes the bottom left quadrant (NB: left means left lung, which is te right side of the EIT
image); - `LAYER_1_MASK` includes only the first 8 rows; - `LAYER_2_MASK` includes only the second set of 8 rows; -
`LAYER_3_MASK` includes only the third set of 8 rows; - `LAYER_4_MASK` includes only the last 8 rows.
"""

import dataclasses
import sys
from dataclasses import InitVar, dataclass
from dataclasses import replace as dataclass_replace
from typing import Literal, Self, TypeVar, overload

import numpy as np

from eitprocessing.datahandling.eitdata import EITData


class PixelMap:  # noqa: D101
    values: np.ndarray

    def update(self, *args, **kwargs) -> Self: ...  # noqa: D102


T = TypeVar("T", np.ndarray, EITData, PixelMap)


@dataclass
class Mask:
    """Mask pixels by selecting or weighing them individually.

    A mask is an array with a value for each pixel. Normally, this value is 0 (or False), 1 (or True), and less commonly
    a value between 0 and 1. When a mask is applied to a dataset, each pixel in that dataset is multiplied by the
    corresponding masking value. Masking is often used to remove specific pixels by multiplying them by 0. Masking can
    also be used to weigh pixels, e.g., for a weighted summation. Masking values that are negative or higher than 1 will
    result in a `ValueError` being raised. You can override this check with `ignore_value_range=True`.

    By default, any masking pixel with the value 0 is converted to `np.nan` ('not a number', NaN). NaN values are
    different from 0, in that 0 means 'this value is small' and NaN means 'there is no value'. E.g., values outside the
    region of interest should generally be NaN, not 0. You can prevent the conversion of 0 values to NaN with
    `zeros_to_nan=False`.

    Masks can be combined by either adding or multiplying them. In either case, the masking values are multiplied to
    create a new mask.

    Example:
    ```python
    assert VENTRAL_MASK + RIGHT_MASK == QUADRANT_1_MASK  # True, quadrant 1 is the ventral part of the right lung
    assert DORSAL_MASK * LEFT_MASK == QUADRANT_4_MASK  # True, quadrant 4 is the dorsal part of the left lung
    ```

    """

    mask: np.ndarray
    description: str | None = None
    zeros_to_nan: InitVar[bool] = True
    ignore_value_range: InitVar[bool] = False

    def __post_init__(self, zeros_to_nan: bool, ignore_value_range: bool):
        if not any(np.issubdtype(self.mask.dtype, type_) for type_ in (np.integer, np.floating, np.bool_)):
            msg = f"Mask data type should be a bool, int or float, not {self.mask.dtype}."

        if not ignore_value_range and self.is_weighted and (np.any(self.mask > 1) or np.any(self.mask < 0)):
            msg = "One or more mask values fall outside the range 0 to 1."
            exc = ValueError(msg)
            if sys.version_info >= (3, 11):
                exc.add_note("Mask values should normally be a boolean value or a number from 0 to 1.")
                exc.add_note(
                    "In case you need a mask with values outside this range, "
                    "provide `ignore_value_range=True` when initializing a Mask."
                )

        if zeros_to_nan:
            new_mask = np.copy(self.mask)
            new_mask[new_mask == 0.0] = np.nan
            object.__setattr__(self, "mask", new_mask)

    @overload
    def apply(self, data: np.ndarray, label: Literal[None] = ...) -> np.ndarray: ...

    @overload
    def apply(self, data: EITData, label: str | None = ...) -> EITData: ...

    @overload
    def apply(self, data: PixelMap, label: str | None = ...) -> PixelMap: ...

    def apply(self, data, label=None):
        """Apply mask to data, returning a copy of the object with values masked.

        Data can be a numpy array, an EITData object or PixelMap object. In case of an EITData object, the mask will be
        applied to the `pixel_impedance` attribute. In case of a PixelMap, the mask will be applied to the `values`
        attribute.

        The input data can have any dimension. The mask is applied to the last two dimensions. The size of the last two
        dimensions must match the size of the dimensions of the mask, and will generally (but do not have to) have the
        length 32.

        The function returns the same data type as `data`. In case of `EITData` or `PixelMap` data, the object will have
        the provided label, or the original data label if none is provided.
        """

        def transform_and_mask(data: np.ndarray) -> np.ndarray:
            transform_data = np.copy(data)
            for _ in range(transform_data.ndim - 2):
                # add dimensions, to allow proper multiplication of multi-dimensional array
                transform_data = transform_data[None, ...]

            return transform_data * self.mask

        match data:
            case np.ndarray():
                return transform_and_mask(data)
            case EITData():
                return dataclass_replace(
                    data, pixel_impedance=transform_and_mask(data.pixel_impedance), label=label or data.label
                )
            case PixelMap():
                return dataclass_replace(data, values=transform_and_mask(data.values), label=label or data.label)
            case _:
                msg = f"Data should be an array, or EITData or PixelMap object, not {type(data)}."
                raise TypeError(msg)

    @property
    def is_weighted(self) -> bool:
        """Whether the mask multiplies any pixels with a number other than 0 or 1."""
        return set(np.unique(self.mask.astype(float)).tolist()) != {0.0, 1.0}

    def __add__(self, other: Self) -> Self:
        """Combine masks by multiplying masking values."""
        return dataclasses.replace(self, mask=self.mask * other.mask)

    __mul__ = __add__


VENTRAL_MASK = Mask(np.concat([np.ones((16, 32)), np.zeros((16, 32))], axis=0))
DORSAL_MASK = Mask(np.concat([np.zeros((16, 32)), np.ones((16, 32))], axis=0))
RIGHT_MASK = Mask(np.concat([np.ones((32, 16)), np.zeros((32, 16))], axis=1))
LEFT_MASK = Mask(np.concat([np.zeros((32, 16)), np.ones((32, 16))], axis=1))
QUADRANT_1_MASK = VENTRAL_MASK + RIGHT_MASK
QUADRANT_2_MASK = VENTRAL_MASK + LEFT_MASK
QUADRANT_3_MASK = DORSAL_MASK + RIGHT_MASK
QUADRANT_4_MASK = DORSAL_MASK + LEFT_MASK
LAYER_1_MASK = Mask(np.concat([np.ones((8, 32)), np.zeros((24, 32))], axis=0))
LAYER_2_MASK = Mask(np.concat([np.zeros((8, 32)), np.ones((8, 32)), np.zeros((16, 32))], axis=0))
LAYER_3_MASK = Mask(np.concat([np.zeros((16, 32)), np.ones((8, 32)), np.zeros((8, 32))], axis=0))
LAYER_4_MASK = Mask(np.concat([np.zeros((24, 32)), np.ones((8, 32))], axis=0))
