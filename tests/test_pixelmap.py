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
    PlotParameters,
    SignedPendelluftMap,
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


def test_add():
    pm1 = PerfusionMap([[0, 1, 2, 3]], plot_parameters={"absolute": True})
    pm1_add = pm1 + 3
    assert np.array_equal(pm1_add.values, np.array([[3, 4, 5, 6]]))
    assert isinstance(pm1_add, PerfusionMap)
    assert pm1_add.plot_parameters == pm1.plot_parameters

    add_pm1 = 3 + pm1
    assert np.array_equal(pm1_add.values, add_pm1.values)
    assert isinstance(add_pm1, PerfusionMap)
    assert add_pm1.plot_parameters == pm1.plot_parameters

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
    assert pm1_sub.plot_parameters == pm1.plot_parameters

    sub_pm1 = 3 - pm1
    assert np.array_equal(sub_pm1.values, [[3, 2, 1, 0]])
    assert isinstance(sub_pm1, PerfusionMap)
    assert sub_pm1.plot_parameters == pm1.plot_parameters

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
    assert pm1_mul.plot_parameters == pm1.plot_parameters

    mul_pm1 = 3 * pm1
    assert np.array_equal(pm1_mul.values, mul_pm1.values)
    assert isinstance(mul_pm1, PerfusionMap)
    assert mul_pm1.plot_parameters == pm1.plot_parameters

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
    assert pm1_div.plot_parameters == pm1.plot_parameters

    with pytest.warns(UserWarning, match="Dividing by 0 will result in `np.nan`"):
        div_pm1 = 3 / pm1
    assert np.array_equal(div_pm1.values, [[np.nan, 3, 3 / 2, 1]], equal_nan=True)
    assert isinstance(div_pm1, PerfusionMap)
    assert div_pm1.plot_parameters == pm1.plot_parameters

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
    assert mean_all.plot_parameters.colorbar is False

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
