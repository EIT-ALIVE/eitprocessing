from dataclasses import FrozenInstanceError

import frozendict
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm, Colormap, Normalize
from matplotlib.ticker import PercentFormatter, ScalarFormatter

from eitprocessing.datahandling.pixelmap import (
    DifferenceMap,
    ODCLMap,
    PerfusionMap,
    PixelMap,
    PlotParameters,
    TIVMap,
)
from eitprocessing.plotting.helpers import AbsolutePercentFormatter, AbsoluteScalarFormatter


def test_init_values():
    pm = PixelMap(np.ones((3, 3)))
    assert isinstance(pm.values, np.ndarray)

    pm = PixelMap([[0]])
    assert isinstance(pm.values, np.ndarray)

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        pm.values[0, 0] = 1

    with pytest.raises(ValueError, match="`values` should have 2 dimensions, not 1"):
        PixelMap([0])

    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'values'"):
        pm.values = [[1]]


def test_init_plot_parameters():
    pm0 = PixelMap([[0]])
    pm1 = PixelMap([[0]], plot_parameters={})

    assert pm0.plot_parameters == pm1.plot_parameters
    assert pm0.plot_parameters is not pm1.plot_parameters

    pm2 = PixelMap([[0]], plot_parameters=PlotParameters(facecolor="white"))
    pm3 = PixelMap([[0]], plot_parameters={"facecolor": "white"})

    assert pm2.plot_parameters != pm0.plot_parameters
    assert pm2.plot_parameters == pm3.plot_parameters

    pm4 = pm0.update(plot_parameters=PlotParameters(facecolor="white"))
    pm5 = pm0.update(plot_parameters={"facecolor": "white"})

    assert pm4.plot_parameters != pm0.plot_parameters
    assert pm4.plot_parameters == pm2.plot_parameters
    assert pm4.plot_parameters == pm5.plot_parameters

    pm6 = pm2.update(plot_parameters={"cmap": "inferno"})
    pm7 = pm2.update(plot_parameters=PlotParameters(cmap="inferno"))
    pm8 = pm2.update(plot_parameters=pm2.plot_parameters.update(cmap="inferno"))

    # when passing a dict as plot_parameters, plot_parameters is *updated*, not *replaced*
    # pm6 works the same as pm8 internally
    # when passing a PlotParameters object, the entire object is replaced
    # pm7 should probably almost never be used
    assert pm6.plot_parameters != pm7.plot_parameters
    assert pm7.plot_parameters.facecolor != "white"
    assert pm6.plot_parameters == pm8.plot_parameters

    with pytest.raises(TypeError, match=r"PlotParameters.__init__\(\) got an unexpected keyword argument"):
        _ = PixelMap([[0]], plot_parameters={"non_existing": None})

    # normally, colorbar kwargs is empty
    assert pm6.plot_parameters.colorbar_kwargs == {}
    pm9 = PixelMap([[0]], plot_parameters={"colorbar_kwargs": {"key": "value"}})
    assert pm9.plot_parameters.colorbar_kwargs == {"key": "value"}

    # when updating, update colorbar kwargs instead of overwriting it
    pm10 = pm9.update(plot_parameters={"colorbar_kwargs": {"foo": "bar"}})
    assert pm10.plot_parameters.colorbar_kwargs == {"key": "value", "foo": "bar"}

    # overriding existing value
    pm11 = pm9.update(plot_parameters={"colorbar_kwargs": {"key": "another value"}})
    assert pm11.plot_parameters.colorbar_kwargs == {"key": "another value"}


def test_threshold():
    pm = PixelMap(np.reshape(np.arange(-50, 50), (10, 10)))
    pm_threshold = pm.threshold(10)

    assert type(pm) is type(pm_threshold)

    pm = ODCLMap(np.reshape(np.arange(-50, 50), (10, 10)))
    pm_threshold = pm.threshold(10)

    assert np.nanmin(pm_threshold.values) == 10
    non_nan_values = pm_threshold.values[~np.isnan(pm_threshold.values)]
    assert np.all(non_nan_values >= 10)

    pm_abs_threshold = pm.threshold(10, absolute=True)
    assert np.nanmin(pm_abs_threshold.values) == -50

    non_nan_values = pm_abs_threshold.values[~np.isnan(pm_abs_threshold.values)]
    assert np.array_equal(non_nan_values, np.concatenate([np.arange(-50, -9), np.arange(10, 50)]))

    pm_lower_threshold = pm.threshold(10, comparator=np.less)
    assert np.nanmax(pm_lower_threshold.values) == 9
    non_nan_values = pm_lower_threshold.values[~np.isnan(pm_lower_threshold.values)]
    assert np.array_equal(non_nan_values, np.arange(-50, 10))


def test_convert():
    pm0 = PixelMap([[0]])
    pm1 = pm0.convert_to(PixelMap)
    assert pm0 == pm1

    pm2 = pm0.convert_to(PerfusionMap)
    assert pm0 != pm2
    assert isinstance(pm2, PerfusionMap)

    with pytest.raises(TypeError, match=r"`target_type` must be \(a subclass of\) PixelMap"):
        _ = pm0.convert_to(PlotParameters)


def test_pixel_map():
    pm = PixelMap([[0]])

    assert pm.plot_parameters.cmap == "viridis"
    assert pm.plot_parameters.norm == "linear"

    assert isinstance(pm.plot_parameters.colorbar_kwargs, frozendict.frozendict)
    assert pm.plot_parameters.colorbar_kwargs == {}


def test_tiv_map():
    pm = TIVMap([[0]])

    assert isinstance(pm.plot_parameters.cmap, Colormap)
    assert pm.plot_parameters.cmap.name == "Blues_r"

    assert isinstance(pm.plot_parameters.norm, Normalize)
    assert pm.plot_parameters.norm.vmin == 0
    assert pm.plot_parameters.norm.vmax is None

    assert isinstance(pm.plot_parameters.colorbar_kwargs, frozendict.frozendict)
    assert pm.plot_parameters.colorbar_kwargs["label"] == "TIV (a.u.)"


def test_imshow():
    pm = TIVMap(np.reshape(np.arange(100), (10, 10)))
    cm = pm.imshow()
    assert cm.axes == plt.gca()
    assert cm.cmap == pm.plot_parameters.cmap
    axes = cm.figure.get_axes()
    assert len(axes) == 2  # colorbar is also an axes

    assert cm.axes in axes
    colorbar_axes = next(iter(set(axes) - {cm.axes}))

    assert colorbar_axes.get_ylabel() == pm.plot_parameters.colorbar_kwargs["label"]


def test_imshow_norm():
    pm = TIVMap(np.reshape(np.arange(10, 110), (10, 10)))

    cm1 = pm.imshow()
    assert cm1.norm.vmin == 0
    assert cm1.norm.vmax == 109
    assert np.amin(cm1.get_array()) == 10
    assert np.amax(cm1.get_array()) == 109

    cm2 = pm.imshow(vmin=20, vmax=50)
    assert cm2.norm.vmin == 20
    assert cm2.norm.vmax == 50
    assert np.amin(cm2.get_array()) == 10
    assert np.amax(cm2.get_array()) == 109

    cm3 = pm.imshow(normalize=True)
    data = cm3.get_array()
    assert np.amax(data) == 1
    assert np.amin(data) == 10 / 109


def test_plot_extend():
    cmap = plt.colormaps["Greens"]

    # over and under are min and max values of colormap
    assert np.array_equal(cmap.get_under(), cmap(0.0))
    assert np.array_equal(cmap.get_over(), cmap(1.0))

    # colorbar has no extend
    cm0 = PixelMap([[0]], plot_parameters={"cmap": cmap}).imshow()
    assert cm0.colorbar.extend == "neither"

    # with under set to a different color, colorbar has min extend
    cmap.set_under("orange")
    cm1 = PixelMap([[0]], plot_parameters={"cmap": cmap}).imshow()
    assert cm1.colorbar.extend == "min"

    # with under and over set to a different color, colorbar has both extend
    cmap.set_over("red")
    cm2 = PixelMap([[0]], plot_parameters={"cmap": cmap}).imshow()
    assert cm2.colorbar.extend == "both"

    # with over set to a different color, colorbar has max extend
    cmap = plt.colormaps["Greens"]
    cmap.set_over("red")
    cm3 = PixelMap([[0]], plot_parameters={"cmap": cmap}).imshow()
    assert cm3.colorbar.extend == "max"


def test_formatter():
    pm = PixelMap([[-1, 1]])
    cm1 = pm.imshow()

    plt.gcf().canvas.draw()  # an actual draw is required to test ticks

    ticks = cm1.colorbar.get_ticks()
    formatter = cm1.colorbar.formatter

    assert isinstance(formatter, ScalarFormatter)
    assert float(formatter(ticks[0]).replace("\N{MINUS SIGN}", "-")) == -1.0
    assert float(formatter(ticks[-1])) == 1.0

    cm2 = pm.imshow(absolute=True)

    plt.gcf().canvas.draw()  # an actual draw is required to test ticks

    ticks = cm2.colorbar.get_ticks()
    formatter = cm2.colorbar.formatter

    assert isinstance(formatter, AbsoluteScalarFormatter)
    assert float(formatter(ticks[0]).replace("\N{MINUS SIGN}", "-")) == 1.0
    assert float(formatter(ticks[-1])) == 1.0

    cm3 = pm.imshow(percentage=True)

    plt.gcf().canvas.draw()  # an actual draw is required to test ticks

    ticks = cm3.colorbar.get_ticks()
    formatter = cm3.colorbar.formatter

    assert isinstance(formatter, PercentFormatter)
    assert formatter(ticks[0]) == "\N{MINUS SIGN}100%"
    assert formatter(ticks[-1]) == "100%"

    cm4 = pm.imshow(percentage=True, absolute=True)

    plt.gcf().canvas.draw()  # an actual draw is required to test ticks

    ticks = cm4.colorbar.get_ticks()
    formatter = cm4.colorbar.formatter

    assert isinstance(formatter, AbsolutePercentFormatter)
    assert formatter(ticks[0]) == "100%"
    assert formatter(ticks[-1]) == "100%"


def test_centered_norm():
    pm = DifferenceMap([[-10, 1]])
    assert isinstance(pm.plot_parameters.norm, CenteredNorm)

    cm = pm.imshow()
    assert cm.colorbar.norm.vcenter == 0
    assert cm.colorbar.norm.vmin == -10
    assert cm.colorbar.norm.vmax == 10
