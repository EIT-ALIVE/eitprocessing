"""Handling pixel-based data maps in EIT analysis.

This module provides classes for working with pixel-based data representations of EIT data. Besides the base PixelMap
class, several specialized subclasses are defined for specific use cases, each with its own default plotting parameters:

- PixelMap: Base class for representing 2D pixel data with visualization and manipulation capabilities.
- TIVMap: Specialized map for Tidal Impedance Variation (TIV) data.
- ODCLMap: Specialized map for Overdistention and Collapse (ODCL) analysis.
- DifferenceMap: Specialized map for visualizing differences between pixel maps.
- PerfusionMap: Specialized map for perfusion analysis.
- PendelluftMap: Specialized map for positive-only pendelluft values (severity, no direction).
- SignedPendelluftMap: Specialized map for signed pendelluft values (severity and direction/phase).

The subclasses have no additional attributes beyond those of PixelMap, but they provide specific default plotting
parameters. Plotting parameters for each map are managed via the `PixelMapPlotParameters` dataclass, which allows for
flexible configuration of colormap, normalization, colorbar, and other display options. These can be customized per map
type or per instance. Plotting parameters are defined and registered in `eitprocessing.plotting.pixelmap`. Each map
type's default parameters provides appropriate default colormaps and normalizations suitable for their specific use
case, along with methods for visualization and data manipulation.

All classes are immutable to ensure data integrity during analysis pipelines.

`PixelMap` provides an `update` method, which allows for creating a copy of the object with one or more attributes
replaced, similar to `dataclasses.replace()` and `copy.replace()` in Python 3.13 and newer. This is useful for updating
data or configuration in an immutable and chainable way, and supports partial updates of nested configuration (e.g.,
updating only a single plotting parameter).
"""

from __future__ import annotations

import warnings
from dataclasses import KW_ONLY, InitVar, asdict, dataclass, field, replace
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numpy import typing as npt
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from eitprocessing.plotting.pixelmap import PixelMapPlotParameters, PixelMapPlotting

T = TypeVar("T", bound="PixelMap")


@dataclass(frozen=True, init=False)
class PixelMap:
    """Map representing a single value for each pixel.

    For many common cases, specific classes with default plot parameters are available.

    Args:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (dict | None):
            Plotting parameters controlling colormap, normalization, colorbar, and other display options. Accepts both a
            PixelMapPlotParameters instance or a dict, which is converted to PixelMapPlotParameters during
            initialization. Subclasses provide their own defaults.

    Attributes:
        plotting (PixelMapPlotting):
            A utility class for plotting the pixel map with the specified parameters. Provides `imshow()` method to plot
            image.
    """

    values: np.ndarray
    _: KW_ONLY
    label: str | None = None
    plot_parameters: InitVar[PixelMapPlotParameters | dict | None]
    _plot_parameters: PixelMapPlotParameters = field(init=False, repr=False)

    def __init__(
        self,
        values: npt.ArrayLike,
        *,
        label: str | None = None,
        plot_parameters: PixelMapPlotParameters | dict | None = None,
    ):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:  # noqa: PLR2004, ignore hardcoded value
            msg = f"`values` should have 2 dimensions, not {values.ndim}."
            raise ValueError(msg)

        values.flags.writeable = False  # Make the values array immutable
        object.__setattr__(self, "values", values)

        object.__setattr__(self, "label", label)

        if plot_parameters is None:
            plot_parameters = {}
        if isinstance(plot_parameters, dict):
            # subclasses define their own version, so grab that if it exists
            from eitprocessing.plotting.pixelmap import get_pixelmap_plot_parameters

            default_plot_parameters = get_pixelmap_plot_parameters(self)
            plot_parameters = default_plot_parameters.update(**plot_parameters)

        object.__setattr__(self, "_plot_parameters", plot_parameters)

    @property
    def plotting(self) -> PixelMapPlotting:
        """A utility class for plotting the pixel map with the specified parameters."""
        from eitprocessing.plotting.pixelmap import PixelMapPlotting

        return PixelMapPlotting(self)

    def threshold(
        self,
        threshold: npt.ArrayLike,
        *,
        comparator: Callable = np.greater_equal,
        absolute: bool = False,
        keep_sign: bool = False,
        fill_value: float = np.nan,
        **return_attrs: dict | None,
    ) -> Self:
        """Threshold the pixel map values.

        This method applies a threshold to the pixel map values, setting values that do not meet the threshold condition
        to a specified fill value. The comparison is done using the provided comparator function (default is `>=`).

        If `absolute` is True, the threshold is applied to the absolute values of the pixel map. If `keep_sign` is True,
        the sign of the original pixel values is retained when filling with the `fill_value`. Otherwise, the fill value
        is applied uniformly.

        The `threshold` method returns a new instance of the same class with the modified values. Other attributes of
        the returned object can be set using keyword arguments.

        Args:
            threshold (float): The threshold value.
            comparator (Callable): A function that compares pixel values against the threshold.
            absolute (bool): If True, apply the threshold to the absolute values of the pixel map.
            keep_sign (bool): If True, retain the sign of the original values when filling.
            fill_value (float): The value to set for pixels that do not meet the threshold condition.
            **return_attrs (dict | None): Additional attributes to pass to the new PixelMap instance.

        Returns:
            Self: A new object instance with the thresholded values.
        """
        compare_values = np.abs(self.values) if absolute else self.values
        sign = np.sign(self.values) if keep_sign else 1.0
        new_values = np.where(comparator(compare_values, threshold), self.values, fill_value * sign)

        return_attrs = return_attrs or {}
        return self.update(values=new_values, **return_attrs)

    def convert_to(self, target_type: type[T], *, keep_attrs: bool = False, **kwargs: dict) -> T:
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
            data.pop("_plot_parameters")

        data.update(kwargs)

        return target_type(**data)

    def __replace__(self, /, **changes) -> Self:
        """Return a copy of the of the PixelMap instance replacing attributes.

        Similar to dataclass.replace(), but with special handling of `plot_parameters`. When `plot_parameters` is
        provided as a dict, it updates the existing `plot_parameters` instead of replacing them completely.

        Args:
            **changes: New values for attributes to replace.

        Returns:
            Self: A new instance with the replaced attributes.
        """
        if "plot_parameters" not in changes:
            changes["plot_parameters"] = self._plot_parameters
        elif isinstance(changes["plot_parameters"], dict):
            changes["plot_parameters"] = self._plot_parameters.update(**changes["plot_parameters"])

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


@dataclass(frozen=True, init=False)
class ODCLMap(PixelMap):
    """Pixel map representing normalized overdistention and collapse.

    Values between -1 and 0 represent collapse (-100% to 0%), while values between 0 and 1 represent overdistention (0%
    to 100%).
    """


@dataclass(frozen=True, init=False)
class DifferenceMap(PixelMap):
    """Pixel map representing the difference between two pixel maps.

    The normalization is centered around zero, with positive values indicating an increase and negative values
    indicating a decrease in the pixel values compared to a reference map. If the values are all expected to be
    positive, converting to a normal `PixelMap` instead.
    """


@dataclass(frozen=True, init=False)
class PerfusionMap(PixelMap):
    """Pixel map representing perfusion values.

    Values represent perfusion, where higher values indicate better perfusion. The values are expected to be
    non-negative, with 0 representing no perfusion and higher values representing more perfusion.
    """


@dataclass(frozen=True, init=False)
class PendelluftMap(PixelMap):
    """Pixel map representing pendelluft values.

    Values represent pendelluft severity as positive values. There is no distinction between pixels with early inflation
    and pixels with late inflation. Alternatively, use SignedPendelluftMap for positive and negative values.
    """


@dataclass(frozen=True, init=False)
class SignedPendelluftMap(PixelMap):
    """Pixel map representing pendelluft values as signed values.

    Values represent pendelluft severity. Negative values indicate pixels that have early inflation (before the global
    inflation starts), while negative values indicate pixels that have late inflation (after the global inflation
    starts).
    """
