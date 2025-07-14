import copy
import sys
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
    PendelluftMap,
    PerfusionMap,
    PixelMap,
    SignedPendelluftMap,
    TIVMap,
)
from eitprocessing.plotting.helpers import AbsolutePercentFormatter, AbsoluteScalarFormatter
from eitprocessing.plotting.pixelmap import (
    PIXELMAP_PLOT_PARAMETERS_REGISTRY,
    PixelMapPlotParameters,
    TIVMapPlotParameters,
)


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


@pytest.mark.skipif(sys.version_info < (3, 13), reason="Requires Python 3.13+")
def test_update():
    pm0 = PixelMap([[0]])

    pm1 = copy.replace(pm0, values=[[1]])
    assert pm1.values == [[1]]
    assert pm1 == pm0.update(values=[[1]])

    pm2 = copy.replace(pm1, label="test")
    assert pm2.values == [[1]]
    assert pm2.label == "test"
    assert pm2 == pm0.update(values=[[1]], label="test")


def test_init_plot_parameters():
    pm0 = PixelMap([[0]])
    pm1 = PixelMap([[0]], plot_parameters={})

    assert pm0._plot_parameters is pm0.plotting.parameters
    assert pm1._plot_parameters is pm1.plotting.parameters
    assert isinstance(pm0.plotting.parameters, PixelMapPlotParameters)
    assert isinstance(pm1.plotting.parameters, PixelMapPlotParameters)

    assert pm0.plotting.parameters == pm1.plotting.parameters
    assert pm0.plotting.parameters is not pm1.plotting.parameters

    pm2 = PixelMap([[0]], plot_parameters=PixelMapPlotParameters(facecolor="white"))
    pm3 = PixelMap([[0]], plot_parameters={"facecolor": "white"})

    assert isinstance(pm2.plotting.parameters, PixelMapPlotParameters)

    assert pm2.plotting.parameters != pm0.plotting.parameters
    assert pm2.plotting.parameters == pm3.plotting.parameters

    with pytest.raises(TypeError, match=r"PlotParameters.__init__\(\) got an unexpected keyword argument"):
        _ = PixelMap([[0]], plot_parameters={"non_existing": None})

    # normally, colorbar kwargs is empty
    assert pm0.plotting.parameters.colorbar_kwargs == {}
    pm3 = PixelMap([[0]], plot_parameters={"colorbar_kwargs": {"key": "value"}})
    assert pm3.plotting.parameters.colorbar_kwargs == {"key": "value"}
    assert isinstance(pm3.plotting.parameters.colorbar_kwargs, frozendict.frozendict)


def test_update_plot_parameters():
    pp0 = PixelMapPlotParameters()
    assert isinstance(pp0, PixelMapPlotParameters)
    pp1 = copy.replace(pp0, cmap="foo")
    assert pp1.cmap == "foo"
    assert pp1 == pp0.update(cmap="foo")

    pm2 = PixelMap([[0]])
    pm3 = PixelMap([[0]], plot_parameters=PixelMapPlotParameters(facecolor="white"))

    pm4 = pm2.update(plot_parameters=PixelMapPlotParameters(facecolor="white"))
    pm5 = pm2.update(plot_parameters={"facecolor": "white"})

    assert pm4.plotting.parameters != pm2.plotting.parameters
    assert pm4.plotting.parameters == pm3.plotting.parameters
    assert pm4.plotting.parameters == pm5.plotting.parameters

    pm6 = pm3.update(plot_parameters={"cmap": "inferno"})
    pm7 = pm3.update(plot_parameters=PixelMapPlotParameters(cmap="inferno"))
    pm8 = pm3.update(plot_parameters=pm3.plotting.parameters.update(cmap="inferno"))

    # when passing a dict as plot_parameters, plot_parameters is *updated*, not *replaced*
    # pm6 works the same as pm8 internally
    # when passing a PlotParameters object, the entire object is replaced
    # pm7 should probably almost never be used
    assert pm6.plotting.parameters != pm7.plotting.parameters
    assert pm7.plotting.parameters.facecolor != "white"
    assert pm6.plotting.parameters == pm8.plotting.parameters

    # when updating, update colorbar kwargs instead of overwriting it
    pm9 = PixelMap([[0]], plot_parameters={"colorbar_kwargs": {"key": "value"}})
    pm10 = pm9.update(plot_parameters={"colorbar_kwargs": {"foo": "bar"}})
    assert pm10.plotting.parameters.colorbar_kwargs == {"key": "value", "foo": "bar"}

    # overriding existing value
    pm11 = pm9.update(plot_parameters={"colorbar_kwargs": {"key": "another value"}})
    assert pm11.plotting.parameters.colorbar_kwargs == {"key": "another value"}


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
        _ = pm0.convert_to(PixelMapPlotParameters)

    # TODO: test kwargs in `convert_to()`


def test_pixel_map():
    pm = PixelMap([[0]])

    assert pm._plot_parameters.cmap == "viridis"
    assert pm._plot_parameters.norm == "linear"

    assert isinstance(pm._plot_parameters.colorbar_kwargs, frozendict.frozendict)
    assert pm._plot_parameters.colorbar_kwargs == {}


def test_tiv_map():
    pm = TIVMap([[0]])

    assert isinstance(pm.plotting.parameters, TIVMapPlotParameters)
    assert isinstance(pm.plotting.parameters.cmap, Colormap)
    assert pm.plotting.parameters.cmap.name == "Blues_r"

    assert isinstance(pm.plotting.parameters.norm, Normalize)
    assert pm.plotting.parameters.norm.vmin == 0
    assert pm.plotting.parameters.norm.vmax is None

    assert isinstance(pm.plotting.parameters.colorbar_kwargs, frozendict.frozendict)
    assert pm.plotting.parameters.colorbar_kwargs["label"] == "TIV (a.u.)"


def test_imshow():
    pm = TIVMap(np.reshape(np.arange(100), (10, 10)))
    cm = pm.plotting.imshow()
    assert cm.axes == plt.gca()
    assert cm.cmap == pm.plotting.parameters.cmap
    axes = cm.figure.get_axes()
    assert len(axes) == 2  # colorbar is also an axes

    assert cm.axes in axes
    colorbar_axes = next(iter(set(axes) - {cm.axes}))

    assert colorbar_axes.get_ylabel() == pm.plotting.parameters.colorbar_kwargs["label"]


def test_imshow_norm():
    pm = TIVMap(np.reshape(np.arange(10, 110), (10, 10)))

    cm1 = pm.plotting.imshow()
    assert cm1.norm.vmin == 0
    assert cm1.norm.vmax == 109
    assert np.amin(cm1.get_array()) == 10
    assert np.amax(cm1.get_array()) == 109

    cm2 = pm.plotting.imshow(vmin=20, vmax=50)
    assert cm2.norm.vmin == 20
    assert cm2.norm.vmax == 50
    assert np.amin(cm2.get_array()) == 10
    assert np.amax(cm2.get_array()) == 109

    cm3 = pm.plotting.imshow(normalize=True)
    data = cm3.get_array()
    assert np.amax(data) == 1
    assert np.amin(data) == 10 / 109


def test_plot_extend():
    cmap = plt.colormaps["Greens"]

    # over and under are min and max values of colormap
    assert np.array_equal(cmap.get_under(), cmap(0.0))
    assert np.array_equal(cmap.get_over(), cmap(1.0))

    # colorbar has no extend
    cm0 = PixelMap([[0]], plot_parameters={"cmap": cmap}).plotting.imshow()
    assert cm0.colorbar.extend == "neither"

    # with under set to a different color, colorbar has min extend
    cmap.set_under("orange")
    cm1 = PixelMap([[0]], plot_parameters={"cmap": cmap}).plotting.imshow()
    assert cm1.colorbar.extend == "min"

    # with under and over set to a different color, colorbar has both extend
    cmap.set_over("red")
    cm2 = PixelMap([[0]], plot_parameters={"cmap": cmap}).plotting.imshow()
    assert cm2.colorbar.extend == "both"

    # with over set to a different color, colorbar has max extend
    cmap = plt.colormaps["Greens"]
    cmap.set_over("red")
    cm3 = PixelMap([[0]], plot_parameters={"cmap": cmap}).plotting.imshow()
    assert cm3.colorbar.extend == "max"


def test_formatter():
    pm = PixelMap([[-1, 1]])
    cm1 = pm.plotting.imshow()

    plt.gcf().canvas.draw()  # an actual draw is required to test ticks

    ticks = cm1.colorbar.get_ticks()
    formatter = cm1.colorbar.formatter

    assert isinstance(formatter, ScalarFormatter)
    assert float(formatter(ticks[0]).replace("\N{MINUS SIGN}", "-")) == -1.0
    assert float(formatter(ticks[-1])) == 1.0

    cm2 = pm.plotting.imshow(absolute=True)

    plt.gcf().canvas.draw()  # an actual draw is required to test ticks

    ticks = cm2.colorbar.get_ticks()
    formatter = cm2.colorbar.formatter

    assert isinstance(formatter, AbsoluteScalarFormatter)
    assert float(formatter(ticks[0]).replace("\N{MINUS SIGN}", "-")) == 1.0
    assert float(formatter(ticks[-1])) == 1.0

    cm3 = pm.plotting.imshow(percentage=True)

    plt.gcf().canvas.draw()  # an actual draw is required to test ticks

    ticks = cm3.colorbar.get_ticks()
    formatter = cm3.colorbar.formatter

    assert isinstance(formatter, PercentFormatter)
    assert formatter(ticks[0]) == "\N{MINUS SIGN}100%"
    assert formatter(ticks[-1]) == "100%"

    cm4 = pm.plotting.imshow(percentage=True, absolute=True)

    plt.gcf().canvas.draw()  # an actual draw is required to test ticks

    ticks = cm4.colorbar.get_ticks()
    formatter = cm4.colorbar.formatter

    assert isinstance(formatter, AbsolutePercentFormatter)
    assert formatter(ticks[0]) == "100%"
    assert formatter(ticks[-1]) == "100%"


def test_centered_norm():
    pm = DifferenceMap([[-10, 1]])
    assert isinstance(pm.plotting.parameters.norm, CenteredNorm)

    cm = pm.plotting.imshow()
    assert cm.colorbar.norm.vcenter == 0
    assert cm.colorbar.norm.vmin == -10
    assert cm.colorbar.norm.vmax == 10


def test_add():
    pm1 = PerfusionMap([[0, 1, 2, 3]], plot_parameters={"absolute": True})
    pm1_add = pm1 + 3
    assert np.array_equal(pm1_add.values, np.array([[3, 4, 5, 6]]))
    assert isinstance(pm1_add, PerfusionMap)
    assert pm1_add.plotting.parameters == pm1.plotting.parameters

    add_pm1 = 3 + pm1
    assert np.array_equal(pm1_add.values, add_pm1.values)
    assert isinstance(add_pm1, PerfusionMap)
    assert add_pm1.plotting.parameters == pm1.plotting.parameters

    pm2 = TIVMap([[1, 2, 3, 4]])
    pm1_add_pm2 = pm1 + pm2
    assert isinstance(pm1_add_pm2, PixelMap)
    assert np.array_equal(pm1_add_pm2.values, [[1, 3, 5, 7]])

    pm3 = PixelMap([[1, 2, 3]])
    with pytest.raises(ValueError, match=r"Shape of PixelMaps \(self: \(1, 4\), other: \(1, 3\)\) do not match."):
        _ = pm1 + pm3


def test_sub():
    pm1 = PerfusionMap([[0, 1, 2, 3]], plot_parameters={"absolute": True})
    pm1_sub = pm1 - 3
    assert np.array_equal(pm1_sub.values, np.array([[-3, -2, -1, 0]]))
    assert isinstance(pm1_sub, PerfusionMap)
    assert pm1_sub.plotting.parameters == pm1.plotting.parameters

    sub_pm1 = 3 - pm1
    assert np.array_equal(sub_pm1.values, [[3, 2, 1, 0]])
    assert isinstance(sub_pm1, PerfusionMap)
    assert sub_pm1.plotting.parameters == pm1.plotting.parameters

    pm2 = TIVMap([[1, 2, 3, 4]])
    pm1_sub_pm2 = pm1 - pm2
    assert isinstance(pm1_sub_pm2, DifferenceMap)
    assert np.array_equal(pm1_sub_pm2.values, [[-1, -1, -1, -1]])
    pm2_sub_pm1 = pm2 - pm1
    assert isinstance(pm2_sub_pm1, DifferenceMap)
    assert np.array_equal(pm2_sub_pm1.values, [[1, 1, 1, 1]])


def test_mul():
    pm1 = PerfusionMap([[0, 1, 2, 3]], plot_parameters={"absolute": True})
    pm1_mul = pm1 * 3
    assert np.array_equal(pm1_mul.values, np.array([[0, 3, 6, 9]]))
    assert isinstance(pm1_mul, PerfusionMap)
    assert pm1_mul.plotting.parameters == pm1.plotting.parameters

    mul_pm1 = 3 * pm1
    assert np.array_equal(pm1_mul.values, mul_pm1.values)
    assert isinstance(mul_pm1, PerfusionMap)
    assert mul_pm1.plotting.parameters == pm1.plotting.parameters

    pm2 = TIVMap([[1, 2, 3, 4]])
    pm1_mul_pm2 = pm1 * pm2
    assert isinstance(pm1_mul_pm2, PixelMap)
    assert np.array_equal(pm1_mul_pm2.values, [[0, 2, 6, 12]])

    pm2_mul_pm1 = pm2 * pm1
    assert isinstance(pm2_mul_pm1, PixelMap)
    assert np.array_equal(pm1_mul_pm2.values, pm2_mul_pm1.values)


def test_div():
    pm1 = PerfusionMap([[0, 1, 2, 3]], plot_parameters={"absolute": True})
    pm1_div = pm1 / 3
    assert np.array_equal(pm1_div.values, np.array([[0, 1 / 3, 2 / 3, 1]]))
    assert isinstance(pm1_div, PerfusionMap)
    assert pm1_div.plotting.parameters == pm1.plotting.parameters

    with pytest.warns(UserWarning, match="Dividing by 0 will result in `np.nan`"):
        div_pm1 = 3 / pm1
    assert np.array_equal(div_pm1.values, [[np.nan, 3, 3 / 2, 1]], equal_nan=True)
    assert isinstance(div_pm1, PerfusionMap)
    assert div_pm1.plotting.parameters == pm1.plotting.parameters

    pm2 = TIVMap([[1, 2, 3, 4]])
    pm1_div_pm2 = pm1 / pm2
    assert isinstance(pm1_div_pm2, PixelMap)
    assert np.array_equal(pm1_div_pm2.values, [[0, 1 / 2, 2 / 3, 3 / 4]])

    with pytest.warns(UserWarning, match="Dividing by 0 will result in `np.nan`"):
        pm2_div_pm1 = pm2 / pm1
    assert isinstance(pm2_div_pm1, PixelMap)
    assert np.array_equal(pm2_div_pm1.values, [[np.nan, 2, 3 / 2, 4 / 3]], equal_nan=True)


def test_nan():
    # preventing 0-values, because the will lead to extra nan-values when dividing
    pm1 = PendelluftMap(np.reshape(np.arange(1, 101), (10, 10)))
    pm2 = ODCLMap(np.reshape(np.linspace(1, 2, 100), (10, 10)))

    # picking random indices (rows/cols) to be set to np.nan
    pm1_rows, pm1_cols = np.unravel_index(np.random.default_rng().choice(100, size=20, replace=False), (10, 10))
    new_pm1_values = pm1.values.copy()
    new_pm1_values[pm1_rows, pm1_cols] = np.nan
    pm1 = pm1.update(values=new_pm1_values)

    pm2_rows, pm2_cols = np.unravel_index(np.random.default_rng().choice(100, size=20, replace=False), (10, 10))
    new_pm2_values = pm2.values.copy()
    new_pm2_values[pm2_rows, pm2_cols] = np.nan
    pm2 = pm2.update(values=new_pm2_values)

    either_isnan = np.isnan(pm1.values) | np.isnan(pm2.values)

    for pm in (pm1 + pm2, pm2 + pm1, pm1 - pm2, pm2 - pm1, pm1 * pm2, pm2 * pm1, pm1 / pm2, pm2 / pm1):
        assert np.array_equal(np.isnan(pm.values), either_isnan)


def test_mean():
    pm1 = PixelMap([[0, 1, 2, 3]])
    pm2 = PerfusionMap([[1, 2, 3, 4]])
    mean_pm1_pm2 = ODCLMap.from_mean([pm1, pm2])

    assert np.array_equal(mean_pm1_pm2.values, [[0.5, 1.5, 2.5, 3.5]])
    assert isinstance(mean_pm1_pm2, ODCLMap)

    pm3 = PendelluftMap([[np.nan, 3, 4, np.nan]])
    pm4 = DifferenceMap([[np.nan, np.nan, 5, 6]])
    mean_pm3_pm4 = SignedPendelluftMap.from_mean([pm3, pm4])
    assert isinstance(mean_pm3_pm4, SignedPendelluftMap)
    assert np.array_equal(mean_pm3_pm4.values, [[np.nan, 3, 4.5, 6]], equal_nan=True)

    mean_all = PixelMap.from_mean([pm1, pm2, pm3, pm4], label="foo", plot_parameters={"colorbar": False})
    assert isinstance(mean_all, PixelMap)
    assert np.array_equal(mean_all.values, np.array([[0.5, 2, 3.5, 13 / 3]]))
    assert mean_all.label == "foo"
    assert mean_all.plotting.parameters.colorbar is False

    array1 = [[0, 1, 2, 3]]
    array2 = [[2, 3, 4, 5]]
    mean_array1_array2 = TIVMap.from_mean([array1, array2])
    assert isinstance(mean_array1_array2, TIVMap)
    assert np.array_equal(mean_array1_array2.values, [[1, 2, 3, 4]])

    array3 = [[[4, 5, 6, 7]]]
    with pytest.raises(ValueError, match="should have 2 dimensions, not 3"):
        _ = PixelMap.from_mean([array3])

    array4 = [[4, 5, 6, 7], [5, 6, 7, 8]]
    with pytest.raises(ValueError, match="all input arrays must have the same shape"):
        _ = PixelMap.from_mean([array1, array4])


def test_replace_defaults():
    """Test that replacing PixelMapPlotParameters defaults works as expected."""
    pm0 = PixelMap([[0]])
    assert pm0.plotting.parameters.cmap == "viridis"

    PIXELMAP_PLOT_PARAMETERS_REGISTRY[PixelMap] = PixelMapPlotParameters(cmap="plasma")

    pm1 = PixelMap([[0]])
    assert pm0.plotting.parameters.cmap == "viridis"
    assert pm1.plotting.parameters.cmap == "plasma"

    pm2 = TIVMap([[0]])
    assert isinstance(pm2.plotting.parameters.cmap, Colormap)
    assert pm2.plotting.parameters.cmap.name == "Blues_r"

    PIXELMAP_PLOT_PARAMETERS_REGISTRY[TIVMap] = PIXELMAP_PLOT_PARAMETERS_REGISTRY[TIVMap].update(cmap="Greens")

    pm3 = TIVMap([[0]])
    assert pm3.plotting.parameters.cmap == "Greens"
    assert pm3.plotting.parameters.colorbar

    PIXELMAP_PLOT_PARAMETERS_REGISTRY[TIVMap] = PIXELMAP_PLOT_PARAMETERS_REGISTRY[TIVMap].update(colorbar=False)

    pm4 = TIVMap([[0]])
    assert pm4.plotting.parameters.cmap == "Greens"
    assert not pm4.plotting.parameters.colorbar
