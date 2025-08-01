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

Each map type provides appropriate default colormaps and normalizations suitable for
their specific use case, along with methods for visualization and data manipulation.

Plotting parameters for each map are managed via a `PlotParameters` dataclass, which allows for flexible configuration
of colormap, normalization, colorbar, and other display options. These can be customized per map type or per instance.

All classes are immutable to ensure data integrity during analysis pipelines.

Both `PixelMap` and `PlotParameters` provide a `replace` method, which allows you to create a copy of the object with
one or more attributes replaced, similar to `dataclasses.replace()`. This is useful for updating data or configuration
in an immutable and chainable way, and supports partial updates of nested configuration (e.g., updating only a single
plotting parameter).
"""

from __future__ import annotations

import sys
import warnings
from copy import deepcopy
from dataclasses import KW_ONLY, MISSING, asdict, dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar, cast

import matplotlib as mpl
import numpy as np
from frozendict import frozendict
from matplotlib import colorbar
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm, Colormap, LinearSegmentedColormap, Normalize
from matplotlib.ticker import PercentFormatter
from numpy import typing as npt
from typing_extensions import Self

from eitprocessing.plotting.helpers import AbsolutePercentFormatter, AbsoluteScalarFormatter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage

    from eitprocessing.roi import PixelMask

ColorType = str | tuple[float, float, float] | tuple[float, float, float, float] | float | Colormap

T = TypeVar("T", bound="PixelMap")


def _get_zero_norm() -> Normalize:
    return Normalize(vmin=0)


def _get_centered_norm() -> CenteredNorm:
    return CenteredNorm(vcenter=0)


@dataclass(frozen=True, kw_only=True)
class PlotParameters:
    """Configuration parameters for plotting pixel maps.

    This class encapsulates visualization settings used when plotting pixel maps, providing a consistent interface for
    controlling the appearance of plots.

    Attributes:
        cmap (str | Colormap):
            The colormap to use for the plot. Can be a string name of a matplotlib colormap or a Colormap instance.
            Defaults to "viridis".
        norm (str | Normalize):
            The normalization to use for the plot. Can be a string or a matplotlib Normalize instance. Defaults to
            "linear".
        facecolor (ColorType):
            The background color for areas with NaN values. Defaults to "darkgrey".
        colorbar (bool): Whether to display a colorbar. Defaults to True.
        percentage (bool): Whether to display values as percentages. Defaults to False.
        absolute (bool): Whether to use absolute values for plotting. Defaults to False.
        colorbar_kwargs (dict | None): Additional arguments to pass to colorbar creation.
            Defaults to None.
        hide_axes (bool): Whether to hide the plot axes. Defaults to True.
        extra_kwargs (dict): Extra arguments passed to `imshow`. Defaults to an empty dict.
    """

    cmap: str | Colormap = "viridis"
    norm: str | Normalize = "linear"
    facecolor: ColorType = "darkgrey"
    colorbar: bool = True
    percentage: bool = False
    absolute: bool = False
    colorbar_kwargs: frozendict = field(default_factory=frozendict)
    hide_axes: bool = True
    extra_kwargs: frozendict = field(default_factory=frozendict)

    def __post_init__(self):
        for key in ("colorbar_kwargs", "extra_kwargs"):
            default_factory = self.__dataclass_fields__[key].default_factory
            default_value = default_factory() if default_factory is not MISSING else None

            # tell type checker that this is definitely a frozendict
            default_value = cast("frozendict", default_value)

            merged = default_value | (getattr(self, key) or {})

            object.__setattr__(self, key, merged)

    def __replace__(self, /, **changes) -> Self:
        """Return a copy of the of the PlotParameters instance replacing attributes.

        Similar to dataclass.replace(), but with special handling of `colorbar_kwargs`. `colorbar_kwargs` is updated
        with the provided dictionary, rather than replaced.

        Args:
            **changes: New values for attributes to replace.

        Returns:
            Self: A new instance with the replaced attributes.
        """
        if "colorbar_kwargs" in changes:
            changes["colorbar_kwargs"] = self.colorbar_kwargs | changes["colorbar_kwargs"]

        return replace(self, **changes)

    update = __replace__


@dataclass(frozen=True, init=False)
class PixelMap:
    """Map representing a single value for each pixel.

    At initialization, values are conveted to a 2D numpy array of floats. The values are immutable after initialization,
    meaning that the `values` attribute cannot be changed directly. Instead, use the `replace(...)` method to create a
    copy with new values or label.

    For many common cases, specific classes with default plot parameters are available.

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters):
            Plotting parameters controlling colormap, normalization, colorbar, and other display options. Accepts both a
            PlotParameters instance or a dict, which is converted to PlotParameters during initialization.  Subclasses
            provide their own defaults.
    """

    values: np.ndarray
    _: KW_ONLY
    label: str | None = None
    plot_parameters: PlotParameters = field(default_factory=PlotParameters)
    allows_negative_values: ClassVar[bool] = True

    def __init__(
        self,
        values: npt.ArrayLike,
        *,
        label: str | None = None,
        plot_parameters: PlotParameters | dict | None = None,
        suppress_negative_warning: bool = False,
        suppress_all_nan_warning: bool = False,
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

        if not self.allows_negative_values and not suppress_negative_warning and np.any(values < 0):
            warnings.warn(
                f"{self.__class__.__name__} initialized with negative values, but `allows_negative_values` is False. "
                "This may lead to unexpected behavior in plotting or analysis.",
                UserWarning,
            )

        values.flags.writeable = False  # Make the values array immutable
        object.__setattr__(self, "values", values)

        object.__setattr__(self, "label", label)

        if plot_parameters is None:
            plot_parameters = {}
        if isinstance(plot_parameters, dict):
            # subclasses define their own version, so grab that if it exists
            plot_parameters_class = getattr(self.__class__, "PlotParameters", PlotParameters)
            plot_parameters = plot_parameters_class(**plot_parameters)

        object.__setattr__(self, "plot_parameters", plot_parameters)

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

    def imshow(
        self,
        colorbar: bool | None = None,
        percentage: bool | None = None,
        absolute: bool | None = None,
        colorbar_kwargs: dict | None = None,
        facecolor: ColorType | None = None,
        hide_axes: bool | None = None,
        **kwargs,
    ) -> AxesImage:
        """Display the pixel map using `imshow`.

        This method is a wrapper around `matplotlib.pyplot.imshow` that provides convenient defaults and formatting
        options for displaying pixel maps.

        Plotting parameters are taken from `plot_parameters`, unless overridden by explicit arguments. Any additional
        keyword arguments are merged with `plot_parameters.extra_kwargs` and passed to `matplotlib.pyplot.imshow`.

        If `colorbar` is True, a colorbar is added to the axes. The appearance of the colorbar can be modified using
        `percentage` and `absolute` flags:

        - `percentage=True` displays the colorbar in percentage units (where 1.0 → 100%).
        - `absolute=True` uses the absolute value of the data for color scaling and labeling.

        If the colormap has underflow or overflow colors (e.g., for negative values), the colorbar will extend
        accordingly. Additional arguments can be passed to control or override the appearance of the colorbar via
        `colorbar_kwargs`, which are passed directly to `matplotlib.pyplot.colorbar`.

        If `hide_axes` is True, the axis ticks and labels are hidden (but the axes remain visible to retain background
        styling such as facecolor).

        Additional keyword arguments are passed directly to `imshow`, allowing for full control over image rendering.
        Notably, you can pass an existing matplotlib Axes object using the `ax` keyword argument.

        Args:
            colorbar (bool | None): Whether to display a colorbar.
            percentage (bool | None): Whether to display the colorbar values as a percentage.
            absolute (bool | None): Whether to display the colorbar using absolute values.
            colorbar_kwargs (dict): Additional arguments passed to `matplotlib.pyplot.colorbar`.
            facecolor (ColorType | None):
                Background color for the axes. If None, uses the facecolor of the PixelMap.
            hide_axes (bool | None): Whether to hide the axes ticks and labels.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If not provided, uses the current axes.
            **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.imshow`.

        Returns:
            AxesImage: The image object created by imshow.
        """
        plot_parameters = self.plot_parameters
        colorbar = plot_parameters.colorbar if colorbar is None else colorbar
        percentage = plot_parameters.percentage if percentage is None else percentage
        absolute = plot_parameters.absolute if absolute is None else absolute
        hide_axes = plot_parameters.hide_axes if hide_axes is None else hide_axes

        kwargs = dict(plot_parameters.extra_kwargs | kwargs)
        ax = kwargs.pop("ax", plt.gca())

        kwargs.setdefault("cmap", plot_parameters.cmap)
        norm = kwargs.setdefault("norm", plot_parameters.norm)

        if isinstance(norm, Normalize):
            if norm is plot_parameters.norm:
                # prevent sharing norm between plots if not explicitly set when calling imshow
                kwargs["norm"] = norm = deepcopy(norm)
            vmin = kwargs.pop("vmin", None)
            vmax = kwargs.pop("vmax", None)
            if vmin is not None:
                norm.vmin = vmin
            if vmax is not None:
                norm.vmax = vmax

        cm = ax.imshow(self.values, **kwargs)

        colorbar_kwargs = dict(plot_parameters.colorbar_kwargs | (colorbar_kwargs or {}))

        if colorbar:
            self._create_colorbar(percentage, absolute, colorbar_kwargs, ax, cm)

        if hide_axes:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        ax.set(facecolor=facecolor or plot_parameters.facecolor)
        return cm

    def _create_colorbar(
        self, percentage: bool, absolute: bool, colorbar_kwargs: dict | None, ax: Axes, cm: AxesImage
    ) -> colorbar.Colorbar:
        """Create a colorbar for the pixel map."""
        colorbar_kwargs = dict(colorbar_kwargs or {})
        plot_parameters = self.plot_parameters

        if "format" not in colorbar_kwargs:
            if absolute and percentage:
                colorbar_kwargs["format"] = AbsolutePercentFormatter(xmax=1, decimals=0)
            elif percentage:
                colorbar_kwargs["format"] = PercentFormatter(xmax=1, decimals=0)
            elif absolute:
                colorbar_kwargs["format"] = AbsoluteScalarFormatter()

        if isinstance((cmap := plot_parameters.cmap), Colormap):
            extend_min = not np.all(cmap.get_under() == cmap(0.0))
            extend_max = not np.all(cmap.get_over() == cmap(1.0))

            extend = None
            if extend_min and extend_max:
                extend = "both"
            elif extend_min:
                extend = "min"
            elif extend_max:
                extend = "max"

            colorbar_kwargs.setdefault("extend", extend)

        return plt.colorbar(cm, ax=ax, **colorbar_kwargs or {})

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
            data.pop("plot_parameters")

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
        plot_parameters = self.plot_parameters
        if "plot_parameters" in changes and isinstance(changes["plot_parameters"], dict):
            changes["plot_parameters"] = plot_parameters.update(**changes["plot_parameters"])

        changes = {"label": None} | changes  # sets label to None if not provided

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
    """Pixel map representing the tidal impedance variation or amplitude.

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True, kw_only=True)
    class PlotParameters(PlotParameters):
        """Configuration parameters for plotting TIV maps.

        The default configuration uses:

        - The 'Blues' colormap reversed (dark blue is no ventilation, lighter blue/white is more/most ventilation)
        - A zero-based normalization starting at 0 for no TIV
        - Default colorbar label "TIV (a.u.)"
        """

        @staticmethod
        def _get_cmap() -> Colormap:
            _tiv_colormap = mpl.colormaps["Blues"].reversed()
            _tiv_colormap.set_under("purple")
            return _tiv_colormap

        cmap: str | Colormap = field(default_factory=_get_cmap)
        norm: str | Normalize = field(default_factory=_get_zero_norm)
        colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="TIV (a.u.)"))

    plot_parameters: PlotParameters = field(default_factory=PlotParameters)
    allows_negative_values: ClassVar[bool] = False


@dataclass(frozen=True, init=False)
class ODCLMap(PixelMap):
    """Pixel map representing normalized overdistention and collapse.

    Values between -1 and 0 represent collapse (-100% to 0%), while values between 0 and 1 represent overdistention (0%
    to 100%).

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True, kw_only=True)
    class PlotParameters(PlotParameters):
        """Configuration parameters for plotting ODCL maps.

        The default configuration uses:

        - A diverging colormap from white (collapse) through black to dark orange (overdistention)
        - Centered normalization around 0
        - Absolute percentage value formatting for the colorbar
        - Default colorbar label "Collapse/Overdistention (%)"
        """

        cmap: str | Colormap = field(
            default_factory=lambda: LinearSegmentedColormap.from_list("ODCL", ["white", "black", "darkorange"])
        )
        norm: str | Normalize = field(default_factory=lambda: CenteredNorm(vcenter=0, halfrange=1))
        percentage: bool = True
        absolute: bool = True
        colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="Collapse/Overdistention (%)"))

    plot_parameters: PlotParameters = field(default_factory=PlotParameters)


@dataclass(frozen=True, init=False)
class DifferenceMap(PixelMap):
    """Pixel map representing the difference between two pixel maps.

    The normalization is centered around zero, with positive values indicating an increase and negative values
    indicating a decrease in the pixel values compared to a reference map. If the values are all expected to be
    positive, converting to a normal `PixelMap` instead.

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True, kw_only=True)
    class PlotParameters(PlotParameters):
        """Configuration parameters for plotting difference maps.

        The default configuration uses:

        - The 'vanimo' colormap
        - Centered normalization around 0
        - Default colorbar label "Difference"
        """

        cmap: str | Colormap = "vanimo"
        norm: str | Normalize = field(default_factory=_get_centered_norm)
        colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="Difference"))

    plot_parameters: PlotParameters = field(default_factory=PlotParameters)


@dataclass(frozen=True, init=False)
class PerfusionMap(PixelMap):
    """Pixel map representing perfusion values.

    Values represent perfusion, where higher values indicate better perfusion. The values are expected to be
    non-negative, with 0 representing no perfusion and higher values representing more perfusion.

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True, kw_only=True)
    class PlotParameters(PlotParameters):
        """Configuration parameters for plotting perfusion maps.

        The default configuration uses:

        - A gradient colormap black (no perfusion) to red (most perfusion)
        - A zero-based normalization starting at 0 for no perfusion
        - Default colorbar label "Perfusion"
        """

        cmap: str | Colormap = field(
            default_factory=lambda: LinearSegmentedColormap.from_list("Perfusion", ["black", "red"])
        )
        norm: str | Normalize = field(default_factory=_get_zero_norm)
        colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="Perfusion"))

    plot_parameters: PlotParameters = field(default_factory=PlotParameters)
    allows_negative_values: ClassVar[bool] = False


@dataclass(frozen=True, init=False)
class PendelluftMap(PixelMap):
    """Pixel map representing pendelluft values.

    Values represent pendelluft severity as positive values. There is no distinction between pixels with early inflation
    and pixels with late inflation. Alternatively,

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True, kw_only=True)
    class PlotParameters(PlotParameters):
        """Configuration parameters for plotting pendelluft maps.

        The default configuration uses:

        - A gradient colormap black (no pendelluft) to forestgreen (most pendelluft)
        - A zero-based normalization starting at 0 for no pendelluft
        - Default colorbar label "Pendelluft"
        """

        cmap: str | Colormap = field(
            default_factory=lambda: LinearSegmentedColormap.from_list("Perfusion", ["black", "forestgreen"])
        )
        norm: Normalize = field(default_factory=_get_zero_norm)
        colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="Pendelluft"))

    plot_parameters: PlotParameters = field(default_factory=PlotParameters)
    allows_negative_values: ClassVar[bool] = False


@dataclass(frozen=True, init=False)
class SignedPendelluftMap(PixelMap):
    """Pixel map representing pendelluft values as signed values.

    Values represent pendelluft severity. Negative values indicate pixels that have early inflation (before the global
    inflation starts), while negative values indicate pixels that have late inflation (after the global inflation
    starts).

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True, kw_only=True)
    class PlotParameters(PlotParameters):
        """Configuration parameters for plotting signed pendelluft maps.

        The default configuration uses:

        - A diverging colormap from deeppink (early inflation) through black to forestgreen (late inflation)
        - Centered normalization around 0 to properly show the difference between early and late inflation
        - Absolute value formatting for the colorbar
        - Default colorbar label "Pendelluft"
        """

        cmap: str | Colormap = field(
            default_factory=lambda: LinearSegmentedColormap.from_list("Perfusion", ["deeppink", "black", "forestgreen"])
        )
        colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="Pendelluft"))
        absolute: bool = True
        norm: Normalize = field(default_factory=_get_centered_norm)

    plot_parameters: PlotParameters = field(default_factory=PlotParameters)
