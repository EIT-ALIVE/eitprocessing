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

from __future__ import annotations

import re
import sys
import warnings
from dataclasses import InitVar, dataclass, field, replace
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING, TypeVar, overload

import numpy as np
from typing_extensions import Self

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.pixelmap import PixelMap

if TYPE_CHECKING:
    from eitprocessing.plotting.pixelmap import PixelMapPlotConfig, PixelMapPlotting

T = TypeVar("T", np.ndarray, "EITData", "PixelMap")


@dataclass(frozen=True)
class PixelMask:  # noqa: PLW1641
    """Mask pixels by selecting or weighing them individually.

    A mask is a 2D array with a value for each pixel. Most often, this value is NaN (`np.nan`, 'not a number') or 1, and
    less commonly a value in the range 0 to 1. NaN values indicate the pixel is not part of the region of interest,
    e.g., falls outside the functional lung space or is not part of the ventral region of the lung. A value of 1
    indicates the pixel is included in the region of interest. A value between 0 and 1 indicates that the pixel is part
    of the region of interest, but is weighted, e.g., for a weighted summation of pixel values, or because the pixel is
    considered part of multiple regions of interest.

    You can initialize a mask using an array or nested list. At initialization, the mask is converted to a floating
    point numpy array.

    By default, 0-values are converted tot NaN. You can override this behaviour with `keep_zeros=True`. You can
    therefore create a mask by supplying boolean values, where `True` indicates the pixel is part of the region of
    interest (`True` equals 1), and `False` indicates it is not (`False` equals 0, and will by default be converted to
    NaN).

    Since masking is not intended for other operations, masking values that are negative or higher than 1 will result in
    a `ValueError`. You can override this check with `suppress_value_range_error=True`.

    A mask can be applied to any pixel dataset, such as an `EITData` object or a `PixelMap` object. The mask is applied
    to the last two dimensions of the data, which must match the shape of the mask. The mask is applied by multiplying
    each pixel in the dataset by the corresponding masking value. Multiplication by NaN always results in NaN.

    Adding, subtracting and multiplying:
        Masks can be combined by adding, subtracting or multiplying them.

        Adding masks results in a mask that includes
        all pixels that are in either mask. For non-weighted masks this is similar to a union of sets. Weighted pixels
        are added and clipped at 1.

        Subtracting masks results in the pixels that are part of the second mask being removed from the first mask. For
        non-weighted masks this is similar to a set difference. For weighted masks, pixel values are clipped at 0.

        Multiplying masks
        results in a mask that includes only pixels that are in both masks. For non-weighted masks this is similar to an
        intersection of sets.

    Example:
    ```python
    >>> assert VENTRAL_MASK * ANATOMICAL_RIGHT_MASK == QUADRANT_1_MASK
    True  # quadrant 1 is the ventral part of the right lung
    >>> assert DORSAL_MASK * ANATOMICAL_LEFT_MASK == QUADRANT_4_MASK
    True  # quadrant 4 is the dorsal part of the left lung
    ```

    """

    mask: np.ndarray
    plot_config: InitVar[PixelMapPlotConfig]
    label: str | None = None
    keep_zeros: InitVar[bool] = field(default=False, kw_only=True)
    suppress_value_range_error: InitVar[bool] = field(default=False, kw_only=True)
    suppress_zero_conversion_warning: InitVar[bool] = field(default=False, kw_only=True)
    suppress_all_nan_warning: InitVar[bool] = field(default=False, kw_only=True)
    _plot_config: PixelMapPlotConfig = field(init=False, repr=False)

    def __init__(
        self,
        mask: list | np.ndarray,
        *,
        label: str | None = None,
        keep_zeros: bool = False,
        suppress_value_range_error: bool = False,
        suppress_zero_conversion_warning: bool = False,
        suppress_all_nan_warning: bool = False,
        plot_config: PixelMapPlotConfig | dict | None = None,
    ):
        is_boolean_mask = np.array(mask).dtype == bool
        mask = np.array(mask, dtype=float)

        if mask.ndim != 2:  # noqa: PLR2004
            msg = f"Mask should be a 2D array, not {mask.ndim}D."
            raise ValueError(msg)

        all_nan = np.all(np.isnan(mask))
        if (not suppress_all_nan_warning) and all_nan:
            warnings.warn(
                "Mask contains only NaN values. This will create in all-NaN results when applied.",
                UserWarning,
                stacklevel=2,
            )

        if (not all_nan) and (not suppress_value_range_error) and (np.nanmax(mask) > 1 or np.nanmin(mask) < 0):
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
                warnings.warn(msg, UserWarning, stacklevel=2)

            mask[mask == 0] = np.nan

        if plot_config is None:
            plot_config = {}
        if isinstance(plot_config, dict):
            from eitprocessing.plotting import get_plot_config  # noqa: PLC0415

            default_config = get_plot_config(self)
            plot_config = default_config.update(**plot_config)

        object.__setattr__(self, "_plot_config", plot_config)

        mask.flags["WRITEABLE"] = False
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "label", label)

    @property
    def values(self) -> np.ndarray:
        """Alias for `mask`."""
        return self.mask

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the mask."""
        return self.mask.shape

    @property
    def plotting(self) -> PixelMapPlotting:
        """Get the plotting configuration for this mask.

        Returns:
            PixelMapPlotting: The plotting configuration for this mask.
        """
        from eitprocessing.plotting.pixelmap import PixelMapPlotting  # noqa: PLC0415

        return PixelMapPlotting(self)

    def __replace__(self, /, **changes) -> Self:
        """Return a copy of the of the PixelMap instance replacing attributes.

        Similar to dataclass.replace(), but with special handling of `plot_config`. When `plot_config` is
        provided as a dict, it updates the existing `plot_config` instead of replacing them completely.

        Args:
            **changes: New values for attributes to replace.

        Returns:
            Self: A new instance with the replaced attributes.
        """
        if "plot_config" not in changes:
            changes["plot_config"] = self._plot_config
        elif isinstance(changes["plot_config"], dict):
            changes["plot_config"] = self._plot_config.update(**changes["plot_config"])
        label = changes.pop("label", None)
        return replace(self, label=label, **changes)

    update = __replace__
    # TODO: add tests for update

    @overload
    def apply(self, data: np.ndarray) -> np.ndarray: ...

    @overload
    def apply(self, data: EITData, **kwargs) -> EITData: ...

    @overload
    def apply(self, data: PixelMap, **kwargs) -> PixelMap: ...

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
        return self.update(mask=self.mask * other.mask, label=None)

    def __add__(self, other: Self) -> Self:
        """Combine masks by adding masking values.

        Values are clipped at 1, so that the resulting mask does not contain values higher than 1.
        """
        new_mask = np.clip(np.nansum([self.mask, other.mask], axis=0), a_min=None, a_max=1)
        new_mask[new_mask == 0] = np.nan
        return self.update(mask=new_mask, label=None)

    def __sub__(self, other: Self) -> Self:
        mask = (np.nan_to_num(self.mask, nan=0) - np.nan_to_num(other.mask, nan=0)).astype(float).clip(min=0)
        return self.update(mask=mask, keep_zeros=False, label=None, suppress_zero_conversion_warning=True)

    def __eq__(self, other: object) -> bool:
        """Check if two masks are equal.

        Two masks are equal if they have the same shape and the same values. The label is not considered in the
        equality check.
        """
        if not isinstance(other, PixelMask):
            return False
        return self.mask.shape == other.mask.shape and np.array_equal(self.mask, other.mask, equal_nan=True)


def get_geometric_mask(mask: str, shape: tuple[int, int] = (32, 32)) -> PixelMask:
    """Get a geometric mask by name.

    Masks can be generated for appropriates shapes, provided by the shape` argument, a tuple of two integers
    representing the number of rows and columns in the EIT image. The shape has to be divisible in the required number
    of

    The function accepts both full names (e.g., "layer 1") and abbreviations (e.g., "L1").

    The following masks are available:
    - "ventral" or "V": the first half rows of the EIT image.
    - "dorsal" or "D": the last half rows of the EIT image.
    - "anatomical right" or "R": the first half columns of the EIT image.
    - "anatomical left" or "L": the last half columns of the EIT image.
    - "layer 1" or "L1": the first quarter rows of the EIT image.
    - "layer 2" or "L2": the second set of quarter rows of the EIT image.
    - "layer 3" or "L3": the third set of quarter rows of the EIT image.
    - "layer 4" or "L4": the last quarter rows of the EIT image.
    - "quadrant 1" or "Q1": the top right quadrant of the EIT image.
    - "quadrant 2" or "Q2": the top left quadrant of the EIT image.
    - "quadrant 3" or "Q3": the bottom right quadrant of the EIT image.
    - "quadrant 4" or "Q4": the bottom left quadrant of the EIT image.

    Args:
        mask: Name of the geometric mask to retrieve (case-sensitive).
        shape:
            Shape of the EIT image, a tuple of two integers representing the number of rows and columns. Defaults to
            (32, 32).

    Returns:
        PixelMask: The requested geometric mask.

    Raises:
        ValueError: If an unknown mask name is provided.
        ValueError: If the shape is not compatible with the requested mask.
    """

    def _check_dimensions(
        name: str, shape: tuple[int, int], *, height_divisor: int | None = None, width_divisor: int | None = None
    ) -> None:
        total_height, total_width = shape
        if isinstance(height_divisor, int) and total_height % height_divisor != 0:
            msg = (
                f"Shape {shape} is not compatible with a {name} mask. "
                "The height must be a multiple of {height_divisor}."
            )
            raise ValueError(msg)
        if isinstance(width_divisor, int) and total_width % width_divisor != 0:
            msg = (
                f"Shape {shape} is not compatible with a {name} mask. The width must be a multiple of {width_divisor}."
            )
            raise ValueError(msg)

    total_height, total_width = shape
    n_layers_quadrants = 4

    # Replace short names "Qn" with "quadrant n" and "Ln" with "layer n" for n in 1-4.
    mask = re.sub(r"^Q([1-4]{1})$", r"quadrant \1", mask)
    mask = re.sub(r"^L([1-4]{1})$", r"layer \1", mask)

    match mask.split(" "):
        case ["ventral"] | ["V"]:
            _check_dimensions("ventral", shape, height_divisor=2)
            height = total_height // 2
            return PixelMask(
                np.concatenate([np.ones((height, total_width)), np.full((height, total_width), np.nan)], axis=0),
                label="ventral",
            )
        case ["dorsal"] | ["D"]:
            _check_dimensions("dorsal", shape, height_divisor=2)

            # Dorsal mask is flipped version of ventral mask
            ventral_mask = get_geometric_mask("ventral", shape)
            return PixelMask(np.flip(ventral_mask.mask, axis=0), label="dorsal")

        case ["anatomical", "right"] | ["R"]:
            _check_dimensions("anatomical right", shape, width_divisor=2)
            width = total_width // 2
            return PixelMask(
                np.concatenate([np.ones((total_height, width)), np.full((total_height, width), np.nan)], axis=1),
                label="anatomical right",
            )
        case ["anatomical", "left"] | ["L"]:
            _check_dimensions("anatomical left", shape, width_divisor=2)

            # Left mask is a flipped version of right mask
            right_mask = get_geometric_mask("anatomical right", shape)
            return PixelMask(np.flip(right_mask.mask, axis=1), label="anatomical left")

        case ["layer", num_str] if num_str.isdigit() and 1 <= int(num_str) <= n_layers_quadrants:
            num = int(num_str)
            _check_dimensions("layer", shape, height_divisor=4)
            height = total_height // n_layers_quadrants
            return PixelMask(
                np.concatenate(
                    [
                        *[np.full((height, total_width), np.nan)] * (num - 1),
                        np.ones((height, total_width)),
                        *[np.full((height, total_width), np.nan)] * (n_layers_quadrants - num),
                    ]
                ),
                label=mask,
            )

        case ["quadrant", num_str] if num_str.isdigit() and 1 <= int(num_str) <= n_layers_quadrants:
            _check_dimensions("quadrant", shape, height_divisor=2, width_divisor=2)

            # Quadrant masks are where a lateral and ventral/dorsal mask intersect.
            lateral = "anatomical right" if num_str in ["1", "3"] else "anatomical left"
            ventral_dorsal = "ventral" if num_str in ["1", "2"] else "dorsal"
            return (get_geometric_mask(ventral_dorsal, shape) * get_geometric_mask(lateral, shape)).update(label=mask)

        case _:
            msg = f"Unknown mask name: {mask}."
            raise ValueError(msg)
