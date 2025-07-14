"""Visualizing pixel maps.

This module provides methods for visualizing pixel maps, as well as configuration parameters with (editable) defaults.
`PixelMapPlotting` should not be used directly, but rather through the `PixelMap` class, which provides the methods of
`PixelMapPlotting` through the `plotting` property.

The plotting parameters are defined in the `PixelMapPlotParameters` class and its subclasses for specific pixel map
types. PIXELMAP_PLOT_PARAMETERS_REGISTRY is a registry that maps pixel map types to their respective plotting
parameters. At initialization of the `PixelMap` class, the appropriate plotting parameters are copied from the registry.
The `get_pixelmap_plot_parameters` function retrieves the plotting parameters for a specific pixel map instance or type.
The registry can be updated using the `set_pixelmap_plot_parameters` function, which allows for changing the defaults
for specific pixel map types or all types at once. The registry can be reset to hardcoded defaults using the
`reset_pixelmap_plot_parameters`.

Examples:
    >>> from eitprocessing.datahandling.pixelmap import PixelMap, TIVMap
    >>> from eitprocessing.plotting.pixelmap import get_pixelmap_plot_parameters, set_pixelmap_plot_parameters

    # Get default parameters for a pixel map instance
    >>> pixel_map = PixelMap(values=np.random.rand(10, 10))
    >>> params = get_pixelmap_plot_parameters(pixel_map)

    # Update parameters for TIVMap
    >>> set_pixelmap_plot_parameters(TIVMap, cmap="plasma")

    # Reset all parameters to hardcoded defaults
    >>> reset_pixelmap_plot_parameters()
"""

from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, field, fields, replace
from typing import Self, TypeVar, get_type_hints

import matplotlib as mpl
import numpy as np
from frozendict import frozendict
from matplotlib import colorbar
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import CenteredNorm, Colormap, LinearSegmentedColormap, Normalize
from matplotlib.image import AxesImage
from matplotlib.ticker import PercentFormatter

from eitprocessing.datahandling.pixelmap import (
    DifferenceMap,
    ODCLMap,
    PendelluftMap,
    PerfusionMap,
    PixelMap,
    SignedPendelluftMap,
    TIVMap,
)
from eitprocessing.plotting.helpers import AbsolutePercentFormatter, AbsoluteScalarFormatter

ColorType = str | tuple[float, float, float] | tuple[float, float, float, float] | float | Colormap


T = TypeVar("T")


def _get_field_type(field_: Field[T], cls: type) -> type[T]:
    # If using future annotations, resolve string to real type
    type_hints = get_type_hints(cls)
    return type_hints[field_.name]


@dataclass(frozen=True, kw_only=True)
class Parameters:
    """Base class for parameters."""

    def __post_init__(self):
        for field_ in fields(self):
            if _get_field_type(field_, self.__class__) in (dict, frozendict):
                # Convert dict fields to frozendict for immutability
                default_factory = field_.default_factory
                default_value = default_factory() if default_factory is not MISSING else {}
                merged = default_value | (getattr(self, field_.name) or {})
                object.__setattr__(self, field_.name, frozendict(merged))

    def __replace__(self, /, **changes) -> Self:
        """Return a copy of the of the PlotParameters instance replacing attributes.

        Similar to dataclass.replace(), but with special handling of `colorbar_kwargs`. `colorbar_kwargs` is updated
        with the provided dictionary, rather than replaced.

        Args:
            **changes: New values for attributes to replace.

        Returns:
            Self: A new instance with the replaced attributes.
        """
        for field_ in fields(self):
            if _get_field_type(field_, self.__class__) in (dict, frozendict) and field_.name in changes:
                # Instead of replacing the existing with the new dict, merge the changes
                changes[field_.name] = getattr(self, field_.name) | changes[field_.name]

        return replace(self, **changes)

    update = __replace__


def _get_zero_norm() -> Normalize:
    return Normalize(vmin=0)


def _get_centered_norm() -> CenteredNorm:
    return CenteredNorm(vcenter=0)


@dataclass
class PixelMapPlotting:
    """Utility class for plotting pixel maps."""

    pixel_map: PixelMap = field(compare=False, repr=False)

    @property
    def parameters(self) -> "PixelMapPlotParameters":
        """Plotting parameters for the pixel map."""
        return self.pixel_map._plot_parameters  # noqa: SLF001

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
        normalize = self.parameters.normalize if normalize is None else normalize
        colorbar = self.parameters.colorbar if colorbar is None else colorbar
        percentage = self.parameters.percentage if percentage is None else percentage
        absolute = self.parameters.absolute if absolute is None else absolute
        hide_axes = self.parameters.hide_axes if hide_axes is None else hide_axes

        kwargs = dict(self.parameters.extra_kwargs | kwargs)
        ax = kwargs.pop("ax", plt.gca())

        kwargs.setdefault("cmap", self.parameters.cmap)
        norm = kwargs.setdefault("norm", self.parameters.norm)

        if isinstance(norm, Normalize):
            if norm is self.parameters.norm:
                # prevent sharing norm between plots if not explicitly set when calling imshow
                kwargs["norm"] = norm = deepcopy(norm)
            vmin = kwargs.pop("vmin", None)
            vmax = kwargs.pop("vmax", None)
            if vmin is not None:
                norm.vmin = vmin
            if vmax is not None:
                norm.vmax = vmax

        values = self.pixel_map.values
        if normalize:
            values = values / np.nanmax(values, initial=1)

        cm = ax.imshow(values, **kwargs)

        colorbar_kwargs = dict(self.parameters.colorbar_kwargs | (colorbar_kwargs or {}))

        if colorbar:
            self._create_colorbar(percentage, absolute, colorbar_kwargs, ax, cm)

        if hide_axes:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        ax.set(facecolor=facecolor or self.parameters.facecolor)
        return cm

    def _create_colorbar(
        self, percentage: bool, absolute: bool, colorbar_kwargs: dict | None, ax: Axes, cm: AxesImage
    ) -> colorbar.Colorbar:
        """Create a colorbar for the pixel map."""
        colorbar_kwargs = dict(colorbar_kwargs or {})

        if "format" not in colorbar_kwargs:
            if absolute and percentage:
                colorbar_kwargs["format"] = AbsolutePercentFormatter(xmax=1, decimals=0)
            elif percentage:
                colorbar_kwargs["format"] = PercentFormatter(xmax=1, decimals=0)
            elif absolute:
                colorbar_kwargs["format"] = AbsoluteScalarFormatter()

        if isinstance((cmap := self.parameters.cmap), Colormap):
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


@dataclass(frozen=True, kw_only=True)
class PixelMapPlotParameters(Parameters):
    """Configuration parameters for plotting pixel maps.

    This class encapsulates visualization settings used when plotting pixel maps, providing a consistent interface
    for controlling the appearance of plots.

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


@dataclass(frozen=True, kw_only=True)
class TIVMapPlotParameters(PixelMapPlotParameters):
    """Configuration parameters for plotting TIV maps.

    The default configuration uses:

    - The 'Blues' colormap reversed (dark blue is no ventilation, lighter blue/white is more/most ventilation)
    - A zero-based normalization starting at 0 for no TIV
    - Default colorbar label "TIV (a.u.)"
    """

    @staticmethod
    def _get_cmap() -> Colormap:
        tiv_colormap = mpl.colormaps["Blues"].reversed()
        tiv_colormap.set_under("purple")
        return tiv_colormap

    cmap: str | Colormap = field(default_factory=_get_cmap)
    norm: str | Normalize = field(default_factory=_get_zero_norm)
    colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="TIV (a.u.)"))


@dataclass(frozen=True, kw_only=True)
class ODCLMapPlotParameters(PixelMapPlotParameters):
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


@dataclass(frozen=True, kw_only=True)
class DifferenceMapPlotParameters(PixelMapPlotParameters):
    """Configuration parameters for plotting difference maps.

    The default configuration uses:

    - The 'vanimo' colormap
    - Centered normalization around 0
    - Default colorbar label "Difference"
    """

    cmap: str | Colormap = "vanimo"
    norm: str | Normalize = field(default_factory=_get_centered_norm)
    colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="Difference"))


@dataclass(frozen=True, kw_only=True)
class PerfusionMapPlotParameters(PixelMapPlotParameters):
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


@dataclass(frozen=True, kw_only=True)
class PendelluftMapPlotParameters(PixelMapPlotParameters):
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


@dataclass(frozen=True, kw_only=True)
class SignedPendelluftMapPlotParameters(PixelMapPlotParameters):
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


PIXELMAP_PLOT_PARAMETERS_REGISTRY = {
    PixelMap: PixelMapPlotParameters(),
    TIVMap: TIVMapPlotParameters(),
    ODCLMap: ODCLMapPlotParameters(),
    DifferenceMap: DifferenceMapPlotParameters(),
    PerfusionMap: PerfusionMapPlotParameters(),
    PendelluftMap: PendelluftMapPlotParameters(),
    SignedPendelluftMap: SignedPendelluftMapPlotParameters(),
}


def get_pixelmap_plot_parameters(obj: "PixelMap | type[PixelMap]") -> PixelMapPlotParameters:
    """Get the appropriate plot parameters for a given pixel map type.

    Args:
        obj (PixelMap | type[PixelMap]): The pixel map instance or pixel map type for which to get the plot parameters.

    Returns:
        PixelMapPlotParameters: The plot parameters specific to the pixel map type.
    """
    if isinstance(obj, PixelMap):
        cls_ = type(obj)
    elif issubclass(obj, PixelMap):
        cls_ = obj
    else:
        msg = f"Expected PixelMap instance or type, got {type(obj)}"
        raise TypeError(msg)

    return PIXELMAP_PLOT_PARAMETERS_REGISTRY.get(cls_, PixelMapPlotParameters())


def set_pixelmap_plot_parameters(*types, **parameters) -> None:
    """Set or update the plot parameters for specified pixel map types.

    Examples:
        >>> set_pixelmap_plot_parameters(TIVMap, cmap="plasma")
        >>> set_pixelmap_plot_parameters(PendelluftMap, SignedPendelluftMap, colorbar=False, absolute=True)
        >>> set_pixelmap_plot_parameters(cmap="viridis")  # Update all types with new cmap

    """
    if not types:
        types = PIXELMAP_PLOT_PARAMETERS_REGISTRY.keys()

    for type_ in types:
        PIXELMAP_PLOT_PARAMETERS_REGISTRY[type_] = PIXELMAP_PLOT_PARAMETERS_REGISTRY[type_].update(**parameters)


def reset_pixelmap_plot_parameters(*types) -> None:
    """Reset plot parameters to their defaults.

    Resets the plot parameter defaults for the specified pixel map types. If no types are specified, all registered
    pixel map types will be reset to their default parameters.
    """
    if not types:
        types = PIXELMAP_PLOT_PARAMETERS_REGISTRY.keys()
    for type_ in types:
        PIXELMAP_PLOT_PARAMETERS_REGISTRY[type_] = PIXELMAP_PLOT_PARAMETERS_REGISTRY[type_].__class__()
