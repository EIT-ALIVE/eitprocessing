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

from collections.abc import Callable
from copy import deepcopy
from dataclasses import KW_ONLY, MISSING, asdict, dataclass, field, replace
from typing import Self, TypeVar, cast

import matplotlib as mpl
import numpy as np
from frozendict import frozendict
from matplotlib import colorbar
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import CenteredNorm, Colormap, LinearSegmentedColormap, Normalize
from matplotlib.image import AxesImage
from matplotlib.ticker import PercentFormatter
from numpy import typing as npt

from eitprocessing.plotting.helpers import AbsolutePercentFormatter, AbsoluteScalarFormatter

ColorType = str | tuple[float, float, float] | tuple[float, float, float, float] | float | Colormap

T = TypeVar("T", bound="PixelMap")


def _get_zero_norm() -> Normalize:
    return Normalize(vmin=0)


def _get_centered_norm() -> CenteredNorm:
    return CenteredNorm(vcenter=0)


@dataclass(frozen=True)
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
        normalize (bool): Whether to normalize values before plotting. Defaults to False.
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
    normalize: bool = False
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

    def replace(self, **changes):
        if "colorbar_kwargs" in changes:
            changes["colorbar_kwargs"] = self.colorbar_kwargs | changes["colorbar_kwargs"]

        return replace(self, **changes)


@dataclass(frozen=True)
class PixelMap:
    """Map representing a single value for each pixel.

    For many common cases, specific classes with default plot parameters are available.

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters | dict):
            Plotting parameters controlling colormap, normalization, colorbar, and other display options. Accepts both a
            PlotParameters instance or a dict. Subclasses provide their own defaults.
    """

    values: np.ndarray
    _: KW_ONLY
    label: str | None = None

    plot_parameters: PlotParameters | dict = field(default_factory=PlotParameters)

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        values.flags.writeable = False  # Make the values array immutable
        object.__setattr__(self, "values", values)

        if isinstance(self.plot_parameters, dict):
            plot_parameters_class = getattr(self.__class__, "PlotParameters", PlotParameters)
            object.__setattr__(self, "plot_parameters", plot_parameters_class(**self.plot_parameters))

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
        return replace(self, values=new_values, **return_attrs)

    def imshow(
        self,
        colorbar: bool | None = None,
        normalize: bool | None = None,
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

        If `colorbar` is True, a colorbar is added to the axes. If `normalize` is True, the pixel values are scaled
        by their maximum value before plotting. The appearance of the colorbar can be modified using `percentage` and
        `absolute` flags:

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
            normalize (bool | None): Whether to scale by the maximum value.
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
        plot_parameters = cast("PlotParameters", self.plot_parameters)
        normalize = plot_parameters.normalize if normalize is None else normalize
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

        values = self.values
        if normalize:
            values = values / np.nanmax(self.values, initial=1)

        cm = ax.imshow(values, **kwargs)

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
        plot_parameters = cast("PlotParameters", self.plot_parameters)

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

    def replace(self, **changes) -> Self:
        """Return a copy of the of the PixelMap instance replacing attributes.

        Similar to dataclass.replace(), but with special handling of `plot_parameters`. When `plot_parameters` is
        provided as a dict, it updates the existing `plot_parameters` instead of replacing them completely.

        Args:
            **changes: New values for attributes to replace.

        Returns:
            Self: A new instance with the replaced attributes.
        """
        plot_parameters = cast("PlotParameters", self.plot_parameters)
        if "plot_parameters" in changes and isinstance(changes["plot_parameters"], dict):
            changes["plot_parameters"] = plot_parameters.replace(**changes["plot_parameters"])

        return replace(self, **changes)


@dataclass(frozen=True)
class TIVMap(PixelMap):
    """Pixel map representing the tidal impedance variation or amplitude.

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters | dict):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True)
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

    plot_parameters: PlotParameters | dict = field(default_factory=PlotParameters)


@dataclass(frozen=True)
class ODCLMap(PixelMap):
    """Pixel map representing normalized overdistention and collapse.

    Values between -1 and 0 represent collapse (-100% to 0%), while values between 0 and 1 represent overdistention (0%
    to 100%).

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters | dict):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True)
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

    plot_parameters: PlotParameters | dict = field(default_factory=PlotParameters)


@dataclass(frozen=True)
class DifferenceMap(PixelMap):
    """Pixel map representing the difference between two pixel maps.

    The normalization is centered around zero, with positive values indicating an increase and negative values
    indicating a decrease in the pixel values compared to a reference map. If the values are all expected to be
    positive, converting to a normal `PixelMap` instead.

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters | dict):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True)
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


@dataclass(frozen=True)
class PerfusionMap(PixelMap):
    """Pixel map representing perfusion values.

    Values represent perfusion, where higher values indicate better perfusion. The values are expected to be
    non-negative, with 0 representing no perfusion and higher values representing more perfusion.

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters | dict):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True)
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

    plot_parameters: PlotParameters | dict = field(default_factory=PlotParameters)


@dataclass(frozen=True)
class PendelluftMap(PixelMap):
    """Pixel map representing pendelluft values.

    Values represent pendelluft severity as positive values. There is no distinction between pixels with early inflation
    and pixels with late inflation. Alternatively,

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters | dict):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True)
    class PlotParameters(PlotParameters):
        """Configuration parameters for plotting pendelluft maps.

        The default configuration uses:
        - A gradient colormap black (no pendelluft) to forestgreen (most pendelluft)
        - A zero-based normalization starting at 0 for no perfusion
        - Default colorbar label "Pendelluft"
        """

        cmap: str | Colormap = field(
            default_factory=lambda: LinearSegmentedColormap.from_list("Perfusion", ["black", "forestgreen"])
        )
        norm: Normalize = field(default_factory=_get_zero_norm)
        colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="Pendelluft"))

    plot_parameters: PlotParameters | dict = field(default_factory=PlotParameters)


@dataclass(frozen=True)
class SignedPendelluftMap(PixelMap):
    """Pixel map representing pendelluft values as signed values.

    Values represent pendelluft severity. Negative values indicate pixels that have early inflation (before the global
    inflation starts), while negative values indicate pixels that have late inflation (after the global inflation
    starts).

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        plot_parameters (PlotParameters | dict):
            Plotting parameters, with defaults specific to this map type (see `TIVMap.PlotParameters`).
    """

    @dataclass(frozen=True)
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

    plot_parameters: PlotParameters | dict = field(default_factory=PlotParameters)
