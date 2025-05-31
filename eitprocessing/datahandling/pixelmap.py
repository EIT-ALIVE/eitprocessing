from collections.abc import Callable
from dataclasses import KW_ONLY, asdict, dataclass, field, replace
from typing import Self, TypeVar

import matplotlib as mpl
import numpy as np
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


@dataclass(frozen=True)
class PixelMap:
    """Map representing a single value for each pixel.

    Common uses are:

    - compliance map, showing the compliance of each pixel;
    - ODCL map, showing overdistention or collapse values for each pixel;
    - perfusion map, showing perfusion for each pixel.

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        cmap (str | Colormap | None):
            Colormap for the pixel map. Can be a string (name of a colormap) or a Colormap object. Defaults to
            "viridis".
        norm (str | Normalize | None):
            Normalization for the pixel map. Can be a string (name of a normalization) or a Normalize object. Defaults
            to "linear" (which is equivalent to `Normalize()`).
        facecolor (ColorType): Face color, i.e., the background color for NaN values.
    """

    values: np.ndarray
    _: KW_ONLY
    label: str | None = None
    cmap: str | Colormap = "viridis"
    norm: str | Normalize = "linear"
    facecolor: ColorType = "darkgrey"

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        values.flags.writeable = False  # Make the values array immutable
        object.__setattr__(self, "values", values)

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

        The `threshold` method returns a new instance of `PixelMap` with the modified values. Other attributes of the
        returned object can be set using keyword arguments.

        Args:
            threshold (float): The threshold value.
            comparator (Callable): A function that compares pixel values against the threshold.
            absolute (bool): If True, apply the threshold to the absolute values of the pixel map.
            keep_sign (bool): If True, retain the sign of the original values when filling.
            fill_value (float): The value to set for pixels that do not meet the threshold condition.
            **return_attrs (dict | None): Additional attributes to pass to the new PixelMap instance.

        Returns:
            Self: A new PixelMap instance with the thresholded values.
        """
        compare_values = np.abs(self.values) if absolute else self.values
        sign = np.sign(self.values) if keep_sign else 1.0
        new_values = np.where(comparator(compare_values, threshold), self.values, fill_value * sign)

        return_attrs = return_attrs or {}
        return replace(self, values=new_values, **return_attrs)

    def imshow(
        self,
        colorbar: bool = True,
        normalize: bool = False,
        percentage: bool = False,
        absolute: bool = False,
        colorbar_kwargs: dict | None = None,
        facecolor: ColorType | None = None,
        hide_axes: bool = True,
        **kwargs,
    ) -> AxesImage:
        """Display the pixel map using `imshow`.

        This method is a wrapper around `matplotlib.pyplot.imshow` that provides convenient defaults and formatting
        options for displaying pixel maps.

        Unless explicitly overridden, the colormap (`cmap`), normalization (`norm`), and axes background color
        (`facecolor`) are taken from the attributes of the `PixelMap` object.

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
            colorbar (bool): Whether to display a colorbar.
            normalize (bool): Whether to scale by the maximum value.
            percentage (bool): Whether to display the colorbar values as a percentage.
            absolute (bool): Whether to display the colorbar using absolute values.
            colorbar_kwargs (dict): Additional arguments passed to `matplotlib.pyplot.colorbar`.
            facecolor (ColorType | None):
                Background color for the axes. If None, uses the facecolor of the PixelMap.
            hide_axes (bool): Whether to hide the axes ticks and labels.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If not provided, uses the current axes.
            **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.imshow`.

        Returns:
            AxesImage: The image object created by imshow.
        """
        ax = kwargs.pop("ax", plt.gca())

        kwargs.setdefault("cmap", self.cmap)
        norm = kwargs.setdefault("norm", self.norm)

        if isinstance(norm, Normalize):
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

        if colorbar:
            self._create_colorbar(percentage, absolute, colorbar_kwargs, ax, cm)

        if hide_axes:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        ax.set(facecolor=facecolor or self.facecolor)
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

        if isinstance((cmap := self.cmap), Colormap):
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
            data.pop("cmap", None)
            data.pop("norm", None)
            data.pop("facecolor", None)

        data.update(kwargs)

        return target_type(**data)


_tiv_colormap = mpl.colormaps["Blues"].reversed()
_tiv_colormap.set_under("purple")

_odcl_colormap = LinearSegmentedColormap.from_list("ODCL", ["white", "black", "darkorange"])


@dataclass(frozen=True)
class TIVMap(PixelMap):
    """Pixel map representing the tidal impedance variation or amplitude.

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        cmap (str | Colormap | None):
            Colormap for the pixel map. Can be a string (name of a colormap) or a Colormap object. Defaults to
            "viridis".
        norm (str | Normalize | None):
            Normalization for the pixel map. Can be a string (name of a normalization) or a Normalize object. Defaults
            to "linear" (which is equivalent to `Normalize()`).
        facecolor (ColorType): Face color, i.e., the background color for NaN values.
    """

    cmap: str | Colormap = field(default_factory=lambda: _tiv_colormap)

    def imshow(self, *args, **kwargs) -> AxesImage:
        """Display the TIV map using `imshow`.

        This method is a wrapper around `PixelMap.imshow` with default colormap and settings for TIV maps.

        The default colormap transitions from dark blue (low TIV) to white (high TIV) and is purple for negative values.
        `vmin` is set to 0.

        Returns:
            AxesImage: The image object created by `imshow`.
        """
        kwargs.setdefault("vmin", 0)
        kwargs.setdefault("normalize", False)
        kwargs.setdefault("percentage", False)
        kwargs.setdefault("absolute", False)
        return super().imshow(*args, **kwargs)


@dataclass(frozen=True)
class ODCLMap(PixelMap):
    """Pixel map representing normalized overdistention and collapse.

    Values between -1 and 0 represent collapse (-100% to 0%), while values between 0 and 1 represent overdistention (0%
    to 100%).

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        cmap (str | Colormap | None):
            Colormap for the pixel map. Can be a string (name of a colormap) or a Colormap object. Defaults to a custom
            colormap ranging from white (collapse) to black (maximum compliance) and dark orange (overdistention).
        norm (str | Normalize | None):
            Normalization for the pixel map. Can be a string (name of a normalization) or a Normalize object. Defaults
            to `CenteredNorm(vcenter=0, halfrange=1)` (values range from -1 to 1 with 0 at the center).
        facecolor (ColorType): Face color, i.e., the background color for NaN values.
    """

    cmap: str | Colormap = field(default_factory=lambda: _odcl_colormap)
    norm: str | Normalize = field(default_factory=lambda: CenteredNorm(vcenter=0, halfrange=1))

    def imshow(self, *args, **kwargs) -> AxesImage:
        """Display the ODCL map using `imshow`.

        This method is a wrapper around `PixelMap.imshow` with default colormap and settings for ODCL maps.

        The default colormap transitions from white (collapse) to black (maximum compliance) and dark orange
        (overdistention). `percentage` is set to True. `absolute` is set to True. `vmin` and `vmax` are set to -1 and 1,
        respectively.

        Returns:
            AxesImage: The image object created by `imshow`.
        """
        kwargs.setdefault("percentage", True)
        kwargs.setdefault("absolute", True)
        return super().imshow(*args, **kwargs)


@dataclass(frozen=True)
class DifferenceMap(PixelMap):
    """Pixel map representing the difference between two pixel maps.

    The normalization is centered around zero, with positive values indicating an increase and negative values
    indicating a decrease in the pixel values compared to a reference map. If the values are all expected to be
    positive, converting to a normal `PixelMap` instead.

    Attributes:
        values (np.ndarray): 2D array of pixel values.
        label (str | None): Label for the pixel map.
        cmap (str | Colormap | None):
            Colormap for the pixel map. Can be a string (name of a colormap) or a Colormap object. Defaults to "vanimo".
        norm (str | Normalize | None):
            Normalization for the pixel map. Can be a string (name of a normalization) or a Normalize object. Defaults
            to `CenteredNorm(vcenter=0)` (normalizes data symmetrically around zero).
        facecolor (ColorType): Face color, i.e., the background color for NaN values.
    """

    cmap: str | Colormap = "vanimo"
    norm: str | Normalize = field(default_factory=lambda: CenteredNorm(vcenter=0))
