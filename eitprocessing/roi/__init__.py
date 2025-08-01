"""Region of Interest selection and pixel masking.

This module contains tools for the selection of regions of interest and masking pixel data. The central class of this
module is `PixelMask`. Any type of region of interest selection results in a `PixelMask` object. A mask can be applied
to any pixel dataset (EITData, PixelMap) with the same shape.

Several default masks have been predefined. NB: the right side of the patient is to the left side of the EIT image and
vice versa.

- `VENTRAL_MASK` includes only the first 16 rows;
- `DORSAL_MASK` includes only the last 16 rows;
- `ANATOMICAL_RIGHT_MASK` includes only the first 16 columns;
- `ANATOMICAL_LEFT_MASK` includes only the last 16 columns;
- `QUADRANT_1_MASK` includes the top right quadrant;
- `QUADRANT_2_MASK` includes the top left quadrant;
- `QUADRANT_3_MASK` includes the bottom right quadrant;
- `QUADRANT_4_MASK` includes the bottom left quadrant;
- `LAYER_1_MASK` includes only the first 8 rows;
- `LAYER_2_MASK` includes only the second set of 8 rows;
- `LAYER_3_MASK` includes only the third set of 8 rows;
- `LAYER_4_MASK` includes only the last 8 rows.
"""

import dataclasses
import sys
import warnings
from dataclasses import InitVar, dataclass, field
from dataclasses import replace as dataclass_replace
from typing import TypeVar, overload

import numpy as np
from typing_extensions import Self

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.pixelmap import PixelMap

T = TypeVar("T", np.ndarray, EITData, PixelMap)


@dataclass(frozen=True)
class PixelMask:
    """Mask pixels by selecting or weighing them individually.

    A mask is a 2D array with a value for each pixel. Most often, this value is NaN (`np.nan`, 'not a number') or 1, and
    less commonly a value between 0 and 1. NaN values indicate the pixel is not part of the region of interest, e.g.,
    falls outside the functional lung space, or is not part of the ventral region of the lung. A value of 1 indicates
    the pixel is included in the region of interest. A value between 0 and 1 indicates that the pixel is part of the
    region of interest, but is weighted, e.g., for a weighted summation of pixel values, or because the pixel is
    considered part of multiple regions of interest.

    You can initialize a mask using an array or nested list. At initialization, the mask is converted to a floating
    point numpy array.

    By default, 0-values are converted tot NaN. You can override this behaviour with `keep_zeros=True`. You can
    therefore create a mask by supplying boolean values, where `True` indicates the pixel is part of the region of
    interest (`True` equals 1), and `False` indicates it is not (`False` equals 0, and will be converted to NaN).

    Since masking is not intended for other operations, masking values that are negative or higher than 1 will result in
    a `ValueError`. You can override this check with `suppress_value_range_error=True`.

    A mask can be applied to any pixel dataset, such as an `EITData` object or a `PixelMap` object. The mask is applied
    to the last two dimensions of the data, which must match the shape of the mask. The mask is applied by multiplying
    each pixel in the dataset by the corresponding masking value. Multiplication by NaN always results in NaN.

    Masks can be combined by either adding or multiplying them. Adding masks results in a mask that includes all pixels
    that are in either mask. Multiplying masks results in a mask that includes only pixels that are in both masks.

    Example:
    ```python
    >>> assert VENTRAL_MASK * ANATOMICAL_RIGHT_MASK == QUADRANT_1_MASK
    True  # quadrant 1 is the ventral part of the right lung
    >>> assert DORSAL_MASK * ANATOMICAL_LEFT_MASK == QUADRANT_4_MASK
    True  # quadrant 4 is the dorsal part of the left lung
    ```

    """

    mask: np.ndarray
    keep_zeros: InitVar[bool] = field(default=False, kw_only=True)
    suppress_value_range_error: InitVar[bool] = field(default=False, kw_only=True)
    suppress_zero_value_warning: InitVar[bool] = field(default=False, kw_only=True)

    def __init__(
        self,
        mask: list | np.ndarray,
        keep_zeros: bool = False,
        suppress_value_range_error: bool = False,
        suppress_zero_conversion_warning: bool = False,
    ):
        is_boolean_mask = np.array(mask).dtype == bool
        mask = np.array(mask, dtype=float)

        if mask.ndim != 2:  # noqa: PLR2004
            msg = f"Mask should be a 2D array, not {mask.ndim}D."
            raise ValueError(msg)

        if (not suppress_value_range_error) and (np.nanmax(mask) > 1 or np.nanmin(mask) < 0):
            msg = "One or more mask values fall outside the range 0 to 1."
            exc = ValueError(msg)
            if sys.version_info >= (3, 11):
                exc.add_note("Provided values should normally be a boolean value or a number from 0 to 1.")
                exc.add_note(
                    "In case you need a mask with values outside this range, "
                    "provide `suppress_value_range_warning=True` when initializing a Mask."
                )
            raise exc

        if (not keep_zeros) and np.any(mask == 0):
            if (not is_boolean_mask) and not suppress_zero_conversion_warning:
                msg = (
                    "Mask contains 0 values, which will be converted to NaN. "
                    "If you want to keep 0 values, provide `keep_zeros=True` when initializing a Mask. "
                    "If you want to suppress this warning, provide `suppress_value_range_warning=True` "
                    "or provide boolean values as input (only for non-weighted masks)."
                )
                warnings.warn(msg, UserWarning)

            mask[mask == 0] = np.nan

        mask.flags["WRITEABLE"] = False
        object.__setattr__(self, "mask", mask)

    @overload
    def apply(self, data: np.ndarray) -> np.ndarray: ...

    @overload
    def apply(self, data: EITData, **kwargs) -> EITData: ...

    @overload
    def apply(self, data: "PixelMap", **kwargs) -> "PixelMap": ...

    def apply(self, data, **kwargs):
        """Apply pixel mask to data, returning a copy of the object with pixel values masked.

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
            """Transform the mask to ensure it has the correct shape for the given data, and apply the mask.

            The mask is transformed by adding new axes to the beginning of the array, such that the number of dimensions
            match the number of dimensions of the data. The last two dimensions will contain the mask itself. This
            allows the mask to be applied correctly, even if the data has more than two dimensions (e.g., a 3D array
            with shape (time, channels, rows, cols)).
            """
            if self.mask.shape[-2:] != data.shape[-2:]:
                msg = (
                    f"Data shape {data.shape} does not match Mask shape {self.mask.shape}. "
                    "The last two dimensions of the mask and data must match."
                )
                raise ValueError(msg)

            mask = self.mask[tuple([np.newaxis] * (data.ndim - 2)) + (...,)]  # noqa: RUF005
            # TODO: Fix line above when compatibility with Python 3.10 is no longer needed

            return data * mask

        match data:
            case np.ndarray():
                return transform_and_mask(data)
            case EITData():
                return dataclass_replace(data, pixel_impedance=transform_and_mask(data.pixel_impedance), **kwargs)
            case PixelMap():
                return data.update(values=transform_and_mask(data.values), **kwargs)
            case _:
                msg = f"Data should be an array, or EITData or PixelMap object, not {type(data)}."
                raise TypeError(msg)

    @property
    def is_weighted(self) -> bool:
        """Whether the mask multiplies any pixels with a number other than NaN or 1."""
        return not bool(np.all(np.isnan(self.mask) | (self.mask == 1.0)))

    def __mul__(self, other: Self) -> Self:
        """Combine masks by multiplying masking values."""
        return dataclasses.replace(self, mask=self.mask * other.mask)

    def __add__(self, other: Self) -> Self:
        """Combine masks by adding masking values.

        Values are clipped at 1, so that the resulting mask does not contain values higher than 1.
        """
        return dataclasses.replace(self, mask=np.clip(np.nansum([self.mask, other.mask], axis=0), a_min=None, a_max=1))


LAYER_1_MASK = PixelMask(np.concat([np.ones((8, 32)), np.zeros((24, 32))], axis=0))
LAYER_2_MASK = PixelMask(np.concat([np.zeros((8, 32)), np.ones((8, 32)), np.zeros((16, 32))], axis=0))
LAYER_3_MASK = PixelMask(np.concat([np.zeros((16, 32)), np.ones((8, 32)), np.zeros((8, 32))], axis=0))
LAYER_4_MASK = PixelMask(np.concat([np.zeros((24, 32)), np.ones((8, 32))], axis=0))

VENTRAL_MASK = LAYER_1_MASK + LAYER_2_MASK
DORSAL_MASK = LAYER_3_MASK + LAYER_4_MASK

ANATOMICAL_RIGHT_MASK = PixelMask(np.concat([np.ones((32, 16)), np.zeros((32, 16))], axis=1))
ANATOMICAL_LEFT_MASK = PixelMask(np.concat([np.zeros((32, 16)), np.ones((32, 16))], axis=1))

QUADRANT_1_MASK = VENTRAL_MASK * ANATOMICAL_RIGHT_MASK
QUADRANT_2_MASK = VENTRAL_MASK * ANATOMICAL_LEFT_MASK
QUADRANT_3_MASK = DORSAL_MASK * ANATOMICAL_RIGHT_MASK
QUADRANT_4_MASK = DORSAL_MASK * ANATOMICAL_LEFT_MASK
