"""Visualizing pixel maps.

This module provides methods for visualizing pixel maps, as well as configurations with (editable) defaults.
`PixelMapPlotting` should not be used directly, but rather through the `PixelMap` class, which provides the methods of
`PixelMapPlotting` through the `plotting` property.

The plotting configurations are defined in the `PixelMapPlotParameters` class and its subclasses for specific pixel map
types. PIXELMAP_PLOT_PARAMETERS_REGISTRY is a registry that maps pixel map types to their respective plotting
configurations. At initialization of the `PixelMap` class, the appropriate plotting configuration is copied from the
registry. The `get_pixelmap_plot_config` function retrieves the plotting configuration for a specific pixel map instance
or type. The registry can be updated using the `set_pixelmap_plot_config` function, which allows for changing the
defaults for specific pixel map types or all types at once. The registry can be reset to hardcoded defaults using the
`reset_pixelmap_plot_config`.

Examples:
    >>> from eitprocessing.datahandling.pixelmap import PixelMap, TIVMap
    >>> from eitprocessing.plotting.pixelmap import get_pixelmap_plot_config, set_pixelmap_plot_config

    # Get default configuration for a pixel map instance
    >>> pixel_map = PixelMap(values=np.random.rand(10, 10))
    >>> params = get_pixelmap_plot_config(pixel_map)

    # Update configuration for TIVMap
    >>> set_pixelmap_plot_config(TIVMap, cmap="plasma")

    # Reset all configurations to hardcoded defaults
    >>> reset_pixelmap_plot_config()
"""

from copy import deepcopy
from dataclasses import dataclass, field

import matplotlib as mpl
import numpy as np
from frozendict import frozendict
from matplotlib import colorbar
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, CenteredNorm, Colormap, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.image import AxesImage
from matplotlib.ticker import PercentFormatter

from eitprocessing.config import Config
from eitprocessing.datahandling.pixelmap import PixelMap
from eitprocessing.plotting.helpers import AbsolutePercentFormatter, AbsoluteScalarFormatter
from eitprocessing.roi import PixelMask

ColorType = str | tuple[float, float, float] | tuple[float, float, float, float] | float | Colormap


@dataclass
class PixelMapPlotting:
    """Utility class for plotting pixel maps and masks."""

    pixel_map: "PixelMap | PixelMask" = field(compare=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.pixel_map, (PixelMap, PixelMask)):
            msg = f"Expected pixel_map to be of type PixelMap or PixelMask, got {type(self.pixel_map)}."
            raise TypeError(msg)

    @property
    def config(self) -> "PixelMapPlotConfig":
        """Plotting configuration for pixel maps and pixel masks."""
        return self.pixel_map._plot_config  # noqa: SLF001

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
        """Display the pixel map or mask using `imshow`.

        This method is a wrapper around `matplotlib.pyplot.imshow` that provides convenient defaults and formatting
        options for displaying pixel maps and masks.

        Plotting configuration is taken from `plot_config`, unless overridden by explicit arguments. Any additional
        keyword arguments are merged with `plot_config.extra_kwargs` and passed to `matplotlib.pyplot.imshow`.

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
        normalize = self.config.normalize if normalize is None else normalize
        colorbar = self.config.colorbar if colorbar is None else colorbar
        percentage = self.config.percentage if percentage is None else percentage
        absolute = self.config.absolute if absolute is None else absolute
        hide_axes = self.config.hide_axes if hide_axes is None else hide_axes

        kwargs = dict(self.config.extra_kwargs | kwargs)
        ax = kwargs.pop("ax", plt.gca())

        kwargs.setdefault("cmap", self.config.cmap)
        norm = kwargs.setdefault("norm", self.config.norm)

        if isinstance(norm, Normalize):
            if norm is self.config.norm:
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

        colorbar_kwargs = dict(self.config.colorbar_kwargs | (colorbar_kwargs or {}))

        if colorbar:
            self._create_colorbar(percentage, absolute, colorbar_kwargs, ax, cm)

        if hide_axes:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        ax.set(facecolor=facecolor or self.config.facecolor)
        return cm

    def _create_colorbar(
        self, percentage: bool, absolute: bool, colorbar_kwargs: dict | None, ax: Axes, cm: AxesImage
    ) -> colorbar.Colorbar:
        """Create a colorbar for the pixel map or mask."""
        colorbar_kwargs = dict(colorbar_kwargs or {})

        if "format" not in colorbar_kwargs:
            if absolute and percentage:
                colorbar_kwargs["format"] = AbsolutePercentFormatter(xmax=1, decimals=0)
            elif percentage:
                colorbar_kwargs["format"] = PercentFormatter(xmax=1, decimals=0)
            elif absolute:
                colorbar_kwargs["format"] = AbsoluteScalarFormatter()

        if isinstance((cmap := self.config.cmap), Colormap):
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


def _get_zero_norm() -> Normalize:
    return Normalize(vmin=0)


def _get_centered_norm() -> CenteredNorm:
    return CenteredNorm(vcenter=0)


@dataclass(frozen=True, kw_only=True)
class PixelMapPlotConfig(Config):
    """Configuration for plotting pixel maps.

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
class TIVMapPlotConfig(PixelMapPlotConfig):
    """Configuration for plotting TIV maps.

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
class ODCLMapPlotConfig(PixelMapPlotConfig):
    """Configuration for plotting ODCL maps.

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
class DifferenceMapPlotConfig(PixelMapPlotConfig):
    """Configuration for plotting difference maps.

    The default configuration uses:

    - The 'vanimo' colormap
    - Centered normalization around 0
    - Default colorbar label "Difference"
    """

    cmap: str | Colormap = "vanimo"
    norm: str | Normalize = field(default_factory=_get_centered_norm)
    colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="Difference"))


@dataclass(frozen=True, kw_only=True)
class PerfusionMapPlotConfig(PixelMapPlotConfig):
    """Configuration for plotting perfusion maps.

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
class PendelluftMapPlotConfig(PixelMapPlotConfig):
    """Configuration for plotting pendelluft maps.

    The default configuration uses:

    - A gradient colormap black (no pendelluft) to forestgreen (most pendelluft)
    - A zero-based normalization starting at 0 for no perfusion
    - Default colorbar label "Pendelluft"
    """

    cmap: str | Colormap = field(
        default_factory=lambda: LinearSegmentedColormap.from_list("Pendelluft", ["black", "forestgreen"])
    )
    norm: Normalize = field(default_factory=_get_zero_norm)
    colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="Pendelluft"))


@dataclass(frozen=True, kw_only=True)
class SignedPendelluftMapPlotConfig(PixelMapPlotConfig):
    """Configuration for plotting signed pendelluft maps.

    The default configuration uses:

    - A diverging colormap from deeppink (early inflation) through black to forestgreen (late inflation)
    - Centered normalization around 0 to properly show the difference between early and late inflation
    - Absolute value formatting for the colorbar
    - Default colorbar label "Pendelluft"
    """

    cmap: str | Colormap = field(
        default_factory=lambda: LinearSegmentedColormap.from_list(
            "SignedPendelluft", ["deeppink", "black", "forestgreen"]
        )
    )
    colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="Pendelluft"))
    absolute: bool = True
    norm: Normalize = field(default_factory=_get_centered_norm)


@dataclass(frozen=True, kw_only=True)
class PixelMaskPlotConfig(PixelMapPlotConfig):
    """Configuration for plotting pixel masks.

    The default configuration uses:

    - A binary colormap (black and white)
    - A zero-based normalization starting at 0 for no mask
    - Default colorbar label "Mask"
    """

    cmap: str | Colormap = field(default_factory=lambda: ListedColormap(["tab:blue"]))
    norm: str | Normalize = field(default_factory=lambda: BoundaryNorm([0, 2], ncolors=1))
    colorbar_kwargs: frozendict = field(default_factory=lambda: frozendict(label="Mask"))
    colorbar: bool = False
