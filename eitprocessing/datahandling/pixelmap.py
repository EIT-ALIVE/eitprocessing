from dataclasses import KW_ONLY, dataclass, field

import matplotlib as mpl
import numpy as np
from matplotlib import colorbar
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import CenteredNorm, Colormap, LinearSegmentedColormap, Normalize
from matplotlib.image import AxesImage
from matplotlib.ticker import PercentFormatter

from eitprocessing.plotting.helpers import AbsolutePercentFormatter, AbsoluteScalarFormatter

ColorType = str | tuple[float, float, float] | tuple[float, float, float, float] | float | Colormap


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
        cmap (str | Colormap | None): Colormap for the pixel map.
        norm (str | Normalize | None): Normalization for the pixel map.
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

        Args:
            colorbar (bool): Whether to display a colorbar.
            normalize (bool): Whether to scale by the maximum value.
            percentage (bool): Whether to display the colorbar values as a percentage.
            absolute (bool): Whether to display the colorbar using absolute values.
            colorbar_kwargs (dict): Additional arguments passed to `matplotlib.pyplot.colorbar`.
            facecolor (ColorType | None):
                Background color for the axes. If None, uses the facecolor of the PixelMap.
            hide_axes (bool): Whether to hide the axes ticks and labels.
            **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.imshow`.
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


_tiv_colormap = mpl.colormaps["Blues"].reversed()
_tiv_colormap.set_under("purple")

_odcl_colormap = LinearSegmentedColormap.from_list("ODCL", ["white", "black", "darkorange"])


@dataclass(frozen=True)
class TIVMap(PixelMap):
    """Pixel map representing the tidal impedance variation or amplitude.

    Attributes:
        values (np.ndarray): 2D array of TIV/amplitude values.
        label (str | None): Label for the TIV map.
        cmap (str | Colormap | None):
            Colormap for plotting the TIV map. Defaults to blue transitioning to white, with negative values represented
            in purple.
        norm (str | Normalize | None): Normalization for the TIV map.
        facecolor (ColorType): The background color of the axes, shown for NaN or masked values.
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
        values (np.ndarray): 2D array of ODCL values.
        label (str | None): Label for the ODCL map.
        cmap (str | Colormap | None):
            Colormap for plotting the TIV map. Defaults to white transitioning to black transitioning to orange.
        norm (str | Normalize | None): Normalization for the ODCL map.
        facecolor (ColorType): The background color of the axes, shown for NaN or masked values.
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
