from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, field, fields, replace
from typing import TYPE_CHECKING, Self, TypeVar, get_type_hints

import numpy as np
from frozendict import frozendict
from matplotlib import colorbar
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import CenteredNorm, Colormap, Normalize
from matplotlib.image import AxesImage
from matplotlib.ticker import PercentFormatter

from eitprocessing.plotting.helpers import AbsolutePercentFormatter, AbsoluteScalarFormatter

ColorType = str | tuple[float, float, float] | tuple[float, float, float, float] | float | Colormap


if TYPE_CHECKING:
    from eitprocessing.datahandling.pixelmap import PixelMap

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

    pixel_map: "PixelMap" = field(compare=False, repr=False)

    @property
    def parameters(self) -> "PixelMap.PlotParameters":
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
