"""Handling and visualizing pixel-based data maps in EIT analysis.

This module provides classes for working with pixel-based data representations of EIT data. The main class hierarchy
consists of:

- PixelMap: Base class for representing 2D pixel data with visualization and manipulation capabilities.
- TIVMap: Specialized map for Tidal Impedance Variation (TIV) data.
- ODCLMap: Specialized map for Overdistention and Collapse (ODCL) analysis.
- DifferenceMap: Specialized map for visualizing differences between pixel maps.
- PerfusionMap: Specialized map for perfusion analysis.
- PendelluftMap: Specialized map for positive-only pendelluft values (severity, no direction).
- SignedPendelluftMap: Specialized map for signed pendelluft values (severity and direction/phase).

Plotting configurations for each map are managed via a class inheriting from `PixelMapPlotConfig`, which allows for
flexible configuration of colormap, normalization, colorbar, and other display options. These can be customized per map
type or per instance. `PixelMapPlotConfig` is found in `eitprocessing.plotting.pixelmap`.

All classes are immutable to ensure data integrity during analysis pipelines.

`PixelMap` provides a `replace` method, which allows you to create a copy of the object with one or more attributes
replaced, similar to `dataclasses.replace()` (or `copy.replace() in Python ≥3.13). This is useful for updating data or
configuration in an immutable and chainable way, and supports partial updates of nested configuration (e.g., updating
only a single plotting configuration).

Mathematical Operations:
    `PixelMap` instances support basic mathematical operations (+, -, *, /) with other `PixelMap` instances, arrays, or
    scalar values. The operations are applied element-wise to the underlying values.

    - Addition (+): Returns a `PixelMap` with values added element-wise.

    - Subtraction (-): Returns a `DifferenceMap` with values subtracted element-wise.

    - Multiplication (*): Returns a `PixelMap` with values multiplied element-wise.

    - Division (/): Returns a Pi`xelMap with values divided element-wise. Division by zero results in NaN values with a
      warning.

    When operating with another `PixelMap` of any type, operations typically return the base PixelMap type, except for
    subtraction which returns a DifferenceMap. When operating with scalars or arrays, operations return the same type as
    the original `PixelMap`.

    Note: Some `PixelMap` subclasses (like `TIVMap` and `PerfusionMap`) do not allow negative values. Operations that
    might produce negative values with these maps will display appropriate warnings.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import KW_ONLY, InitVar, asdict, dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar, cast

import numpy as np
from numpy import typing as npt
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from eitprocessing.plotting.pixelmap import PixelMapPlotConfig, PixelMapPlotting
    from eitprocessing.roi import PixelMask


PixelMapT = TypeVar("PixelMapT", bound="PixelMap")


@dataclass(frozen=True, init=False)
class PixelMap:
    """Map representing a single value for each pixel.

    At initialization, values are conveted to a 2D numpy array of floats. The values are immutable after initialization,
    meaning that the `values` attribute cannot be changed directly. Instead, use the `replace(...)` method to create a
    copy with new values or label.

    For many common cases, specific classes with default plot configurations are available.

    Args:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_config (dict | PixelMapPlotConfig | None):
            Plotting configuration controlling colormap, normalization, colorbar, and other display options. Accepts
            both a PixelMapPlotConfig instance or a dict, which is converted to PixelMapPlotConfig during
            initialization. Subclasses provide their own defaults.

    Attributes:
        allow_negative_values (bool): Whether negative values are allowed in the pixel map.
    """

    values: np.ndarray
    _: KW_ONLY
    label: str | None = None
    plot_config: InitVar[PixelMapPlotConfig]
    _plot_config: PixelMapPlotConfig = field(init=False, repr=False)
    allow_negative_values: ClassVar[bool] = True

    def __init__(
        self,
        values: npt.ArrayLike,
        *,
        label: str | None = None,
        suppress_negative_warning: bool = False,
        suppress_all_nan_warning: bool = False,
        plot_config: PixelMapPlotConfig | dict | None = None,
    ):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:  # noqa: PLR2004, ignore hardcoded value
            msg = f"`values` should have 2 dimensions, not {values.ndim}."
            raise ValueError(msg)

        if not suppress_all_nan_warning and np.all(np.isnan(values)):
            warnings.warn(
                f"{self.__class__.__name__} initialized with all NaN values. "
                "This may lead to unexpected behavior in plotting or analysis.",
                UserWarning,
            )

        if not self.allow_negative_values and not suppress_negative_warning and np.any(values < 0):
            warnings.warn(
                f"{self.__class__.__name__} initialized with negative values, but `allow_negative_values` is False. "
                "This may lead to unexpected behavior in plotting or analysis.",
                UserWarning,
            )

        values.flags.writeable = False  # Make the values array immutable
        object.__setattr__(self, "values", values)

        object.__setattr__(self, "label", label)

        if plot_config is None:
            plot_config = {}
        if isinstance(plot_config, dict):
            from eitprocessing.plotting.pixelmap import get_pixelmap_plot_config

            default_config = get_pixelmap_plot_config(self)
            plot_config = default_config.update(**plot_config)

        object.__setattr__(self, "_plot_config", plot_config)

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the pixel map values."""
        return self.values.shape

    def normalize(
        self,
        *,
        mode: Literal["zero-based", "symmetric", "maximum", "reference"] = "zero-based",
        reference: float | None = None,
        **kwargs,
    ) -> Self:
        """Normalize the pixel map values.

        Creates a copy of the pixel map with normalized values. Four normalization modes are available.

        - "zero-based" (default): normalizes the values to the range [0, 1] by subtracting the minimum value and
          dividing by the new maximum value. This ensures the lowest resulting value is 0 and the highest resulting
          value is 1.
        - "symmetric": divides the values by the maximum absolute value, resulting in a range of [-1, 1].
        - "maximum": divides the values by the maximum value. If all values are positive, this normalizes them to the
          range [0, 1] without shifting the minimum value to zero. The sign of values does not change. If negative
          values are present, the result may extend below -1.
        - "reference": is similar to "maximum", except it divides by a user-defined reference value. A reference value
          must be provided. Resulting values can fall outside the range [-1, 1].

        NaN values are ignored when normalizing. All-NaN pixel maps results in a ValueError.

        Examples:
        ```
        >>> PixelMap([[1, 3, 5]]).normalize()  # Default is zero-based normalization
        PixelMap(values=array([[0. , 0.5, 1. ]]), ...)

        >>> PixelMap([[1, 3, 5]]).normalize(mode="maximum")
        PixelMap(values=array([[0.2, 0.6, 1. ]]), ...)

        >>> PixelMap([[-8, -1, 2]]).normalize()
        PixelMap(values=array([[0. , 0.7, 1. ]]), ...)

        >>> PixelMap([[-8, -1, 2]]).normalize(mode="symmetric")
        PixelMap(values=array([[-1.   , -0.125,  0.25 ]]), ...)

        >>> PixelMap([[-8, -1, 2]]).normalize(mode="reference", reference=4)
        PixelMap(values=array([[-2.  , -0.25,  0.5 ]]), ...)
        ```

        Args:
            mode (Literal["zero-based", "symmetric", "maximum", "reference"]):
                The normalization mode to use. Defaults to "zero-based".
            reference (float | None):
                The reference value to use for normalization in "reference" mode.
            kwargs (dict):
                Additional keyword arguments to pass to the new PixelMap instance.

        Raises:
            ValueError: If an invalid normalization mode is specifief.
            ValueError: If no reference value is provided in "reference" mode.
            ValueError: If a reference value is provided with a mode other than "reference".
            TypeError: If the reference value is not a number.
            ZeroDivisionError:
                If normalization by zero is attempted (either `reference=0`, or the maximum (absolute) value in the
                values is 0).
            ValueError: If normalization by NaN is attempted (either `reference=np.nan`, or all values are NaN).

        Warns:
            UserWarning:
                If normalization by a negative number is attempted (either `reference` is negative, or all values are
                negative). This results in inverting the sign of the values.
        """
        if reference is not None and mode != "reference":
            msg = "`reference` can only be used with `mode='reference'`."
            raise ValueError(msg)

        if mode == "reference":
            if reference is None:
                msg = "`reference` must be provided when `mode='reference'`."
                raise ValueError(msg)
            if not isinstance(reference, (float, int)):
                msg = "`reference` must be a number."
                raise TypeError(msg)
            self._check_normalization_reference(reference)

        reference_: float
        match mode:
            case "symmetric":
                values = self.values
                reference_ = np.nanmax(np.abs(self.values))
            case "zero-based":
                values = self.values - np.nanmin(self.values)
                reference_ = np.nanmax(values)
            case "maximum":
                values = self.values
                reference_ = np.nanmax(values)
            case "reference":
                values = self.values
                reference_ = cast("float", reference)
            case _:
                msg = f"Unknown normalization mode: {mode}"
                raise ValueError(msg)

        self._check_normalization_reference(reference_)

        new_values = values / reference_

        return self.update(values=new_values, **kwargs)

    @staticmethod
    def _check_normalization_reference(reference_: float) -> None:
        if reference_ == 0:
            msg = "Normalization by zero is not allowed."
            exc = ZeroDivisionError(msg)
            if sys.version_info >= (3, 11):
                exc.add_note(
                    "You are either trying to normalize by 0 explicitly, "
                    "or are providing a PixelMap with no non-zero values."
                )
            raise exc
        if np.isnan(reference_):
            msg = "Normalization by NaN is not allowed."
            exc = ValueError(msg)
            if sys.version_info >= (3, 11):
                exc.add_note(
                    "You are either trying to normalize by NaN explicitly, "
                    "or are providing a PixelMap with no non-NaN values."
                )
            raise exc
        if reference_ < 0:
            warnings.warn("Normalization by a negative number may lead to unexpected results.", UserWarning)

    def create_mask_from_threshold(
        self,
        threshold: float,
        *,
        comparator: Callable = np.greater_equal,
        absolute: bool = False,
    ) -> PixelMask:
        """Create a pixel mask from the pixel map based on threshold values.

        The values of the pixel map are compared to the threshold values. By default, the comparator is `>=`
        (`np.greater_equal`), such that the resulting mask is 1.0 where the map values are at least the threshold
        values, and NaN elsewhere. The comparator can be set to any comparison function, e.g.`np.less`, a function from
        the `operator` module or custom function which takes pixel map values array and threshold as arguments, and
        returns a boolean array with the same shape as the array.

        If `absolute` is True, absolute values are compared to the threshold.

        The shape of the pixel mask is the same as the shape of the pixel map.

        Args:
            threshold (float): The threshold value.
            comparator (Callable): A function that compares pixel values against the threshold.
            absolute (bool): If True, apply the threshold to the absolute values of the pixel map.

        Returns:
            PixelMask:
                A PixelMask instance with values 1.0 where comparison is true, and NaN elsewhere.

        Raises:
            TypeError: If `threshold` is not a float or `comparator` is not callable.

        Examples:
        >>> pm = PixelMap([[0.1, 0.5, 0.9]])
        >>> mask = pm.create_mask_from_threshold(0.5)
        PixelMask(mask=array([[nan,  1.,  1.]]))
        >>> mask.apply(pm)
        PixelMap(values=array([[nan, 0.5, 0.9]]), ...)

        >>> mask = pm.create_mask_from_threshold(0.5, comparator=np.less)
        PixelMask(mask=array([[ 1., nan, nan]]))
        """
        if not isinstance(threshold, (float, np.floating, int, np.integer)):
            msg = "`threshold` must be a number."
            raise TypeError(msg)

        if not callable(comparator):
            msg = "`comparator` must be a callable function."
            raise TypeError(msg)

        from eitprocessing.roi import PixelMask

        compare_values = np.abs(self.values) if absolute else self.values
        mask_values = comparator(compare_values, threshold)
        return PixelMask(mask_values)

    @property
    def plotting(self) -> PixelMapPlotting:
        """A utility class for plotting the pixel map with the specified configuration."""
        from eitprocessing.plotting.pixelmap import PixelMapPlotting

        return PixelMapPlotting(self)

    def convert_to(self, target_type: type[PixelMapT], *, keep_attrs: bool = False, **kwargs: dict) -> PixelMapT:
        """Convert the pixel map to (a different subclass of) PixelMap.

        This method allows for converting the pixel map to a `PixelMap` or different subclass of `PixelMap`. The `label`
        attribute is copied by default, but a new label can be provided. Other attributes are not copied by default, but
        can be retained by setting `keep_attrs` to True. Additional keyword arguments can be passed to the new instance.

        Args:
            target_type (type[T]): The target subclass to convert to.
            keep_attrs (bool): If True, retains the attributes of the original pixel map in the new instance.
            **kwargs (dict): Additional keyword arguments to pass to the new instance.

        Returns:
            T: A new instance of the target type with the same values and attributes.
        """
        if not issubclass(target_type, PixelMap):
            msg = "`target_type` must be (a subclass of) PixelMap."
            raise TypeError(msg)

        data = asdict(self)

        if not keep_attrs:
            data.pop("_plot_config")

        data.update(kwargs)

        return target_type(**data)

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

        return replace(self, **changes)

    update = __replace__

    def _validate_other(self, other: npt.ArrayLike | float | PixelMap) -> np.ndarray | float:
        other_values = other.values if isinstance(other, PixelMap) else other

        if isinstance(other, (float, int)):
            return other

        other_values = other.values if isinstance(other, PixelMap) else np.array(other_values)

        if (os := other_values.shape) != (ss := self.values.shape):
            msg = f"Shape of PixelMaps (self: {ss}, other: {os}) do not match."
            raise ValueError(msg)

        return other_values

    def __add__(self, other: npt.ArrayLike | float | PixelMap) -> PixelMap:
        new_values = self.values + self._validate_other(other)
        if isinstance(other, PixelMap):
            return PixelMap(new_values)
        return self.update(values=new_values, label=None)

    __radd__ = __add__

    def __sub__(self, other: npt.ArrayLike | float | PixelMap) -> PixelMap:
        new_values = self.values - self._validate_other(other)
        if isinstance(other, PixelMap):
            return DifferenceMap(new_values)
        return self.update(values=new_values, label=None)

    def __rsub__(self, other: npt.ArrayLike | float | PixelMap) -> PixelMap:
        new_values = -self.values + self._validate_other(other)
        return self.update(values=new_values, label=None)

    def __mul__(self, other: npt.ArrayLike | float | PixelMap) -> PixelMap:
        new_values = self.values * self._validate_other(other)
        if isinstance(other, PixelMap):
            return PixelMap(new_values)
        return self.update(values=new_values, label=None)

    __rmul__ = __mul__

    def __truediv__(self, other: npt.ArrayLike | float | PixelMap) -> PixelMap:
        other_values = self._validate_other(other)
        if isinstance(other_values, np.ndarray) and 0 in other_values:
            warnings.warn("Dividing by 0 will result in `np.nan` value.", UserWarning)

        invalid = np.isnan(self.values) | np.isnan(other_values) | (other_values == 0)
        new_values = np.divide(self.values, other_values, where=~invalid)
        new_values[invalid] = np.nan
        if isinstance(other, PixelMap):
            return PixelMap(new_values)
        return self.update(values=new_values, label=None)

    def __rtruediv__(self, other: npt.ArrayLike | float | PixelMap) -> PixelMap:
        other_values = self._validate_other(other)
        if 0 in self.values:
            warnings.warn("Dividing by 0 will result in `np.nan` value.", UserWarning)
        invalid = np.isnan(self.values) | np.isnan(other_values) | (self.values == 0)
        new_values = np.divide(other_values, self.values, where=~invalid)
        new_values[invalid] = np.nan

        # other should never be a PixelMap, because this method is only called if other doesn't know how to divide,
        # which is does if it is a PixelMap.

        return self.update(values=new_values, label=None)

    @classmethod
    def from_mean(cls, maps: Sequence[npt.ArrayLike | PixelMap], **return_attrs) -> Self:
        """Get a pixel map of the the per-pixel mean of several pixel maps.

        The maps can be 2D numpy arrays, sequences that can be converted to 2D numpy arrays, or PixelMap objects. The
        mean is determined using `np.nanmean`, such that NaN values are ignored.

        Returns the same class as this function was called from. Keyword arguments are passed to the initializer of that
        object.

        Args:
            maps: sequence of maps that contribute to the mean
            **return_attrs: keyword arguments to be passed to the initializer of the return object

        Returns:
            Self: A new instance with the per-pixel mean of the pixel maps.
        """

        def _get_values(map_: npt.ArrayLike | PixelMap) -> np.ndarray:
            if isinstance(map_, PixelMap):
                return map_.values

            map_ = np.asarray(map_, dtype=float)

            if map_.ndim != 2:  # noqa: PLR2004, ignore hardcoded value
                msg = f"Map {map_} should have 2 dimensions, not {map_.ndim}."
                raise ValueError(msg)

            return map_

        stacked = np.stack([_get_values(map_) for map_ in maps])
        mean_values = np.nanmean(stacked, axis=0)
        return cls(values=mean_values, **return_attrs)


@dataclass(frozen=True, init=False)
class TIVMap(PixelMap):
    """Pixel map representing the tidal impedance variation or amplitude."""

    allow_negative_values: ClassVar[bool] = False


@dataclass(frozen=True, init=False)
class ODCLMap(PixelMap):
    """Pixel map representing normalized overdistention and collapse."""


@dataclass(frozen=True, init=False)
class DifferenceMap(PixelMap):
    """Pixel map representing the difference between two pixel maps."""


@dataclass(frozen=True, init=False)
class PerfusionMap(PixelMap):
    """Pixel map representing perfusion values."""

    allow_negative_values: ClassVar[bool] = False


@dataclass(frozen=True, init=False)
class PendelluftMap(PixelMap):
    """Pixel map representing positive-only pendelluft values."""

    allow_negative_values: ClassVar[bool] = False


@dataclass(frozen=True, init=False)
class SignedPendelluftMap(PixelMap):
    """Pixel map representing pendelluft values as signed values."""
