import copy
import sys
import warnings
from dataclasses import FrozenInstanceError

import frozendict
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm, Colormap, Normalize
from matplotlib.ticker import PercentFormatter, ScalarFormatter

from eitprocessing.datahandling.pixelmap import (
    DifferenceMap,
    IntegerMap,
    ODCLMap,
    PendelluftMap,
    PerfusionMap,
    PixelMap,
    SignedPendelluftMap,
    TIVMap,
)
from eitprocessing.plotting import _PLOT_CONFIG_REGISTRY, reset_plot_config, set_plot_config_parameters
from eitprocessing.plotting.helpers import AbsolutePercentFormatter, AbsoluteScalarFormatter
from eitprocessing.plotting.pixelmap import PixelMapPlotConfig


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


def test_shape_property():
    pm = PixelMap([[0]])
    assert pm.shape == (1, 1)

    pm = PixelMap(np.ones((2, 3)))
    assert pm.shape == (2, 3)

    pm = PendelluftMap([[0, 1], [2, 3]])
    assert pm.shape == (2, 2)

    pm = SignedPendelluftMap([[0, -1], [-2, -3]])
    assert pm.shape == (2, 2)


def test_init_negative_values():
    with pytest.warns(UserWarning, match="PendelluftMap initialized with negative values"):
        _ = PendelluftMap([[0, -1], [-2, -3]])

    with warnings.catch_warnings(record=True) as w:
        _ = PendelluftMap([[0, -1], [-2, -3]], suppress_negative_warning=True)  # does not warn
        assert len(w) == 0

    with warnings.catch_warnings(record=True) as w:
        _ = SignedPendelluftMap([[0, -1], [-2, -3]])  # does not warn
        assert len(w) == 0


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

    pp0 = PixelMapPlotConfig()
    pp1 = copy.replace(pp0, cmap="foo")
    assert pp1.cmap == "foo"
    assert pp1 == pp0.update(cmap="foo")


def test_init_plot_config():
    pm0 = PixelMap([[0]])
    pm1 = PixelMap([[0]], plot_config={})

    assert pm0._plot_config is pm0.plotting.config
    assert pm1._plot_config is pm1.plotting.config
    assert isinstance(pm0.plotting.config, PixelMapPlotConfig)
    assert isinstance(pm1.plotting.config, PixelMapPlotConfig)

    assert pm0.plotting.config == pm1.plotting.config
    assert pm0.plotting.config is not pm1.plotting.config

    pm2 = PixelMap([[0]], plot_config=PixelMapPlotConfig(facecolor="white"))
    pm3 = PixelMap([[0]], plot_config={"facecolor": "white"})

    assert pm2.plotting.config != pm0.plotting.config
    assert pm2.plotting.config == pm3.plotting.config

    with pytest.raises(TypeError, match=r"PixelMapPlotConfig.__init__\(\) got an unexpected keyword argument"):
        _ = PixelMap([[0]], plot_config={"non_existing": None})

    pm6 = pm2.update(plot_config={"cmap": "inferno"})

    # normally, colorbar kwargs is empty
    assert pm6.plotting.config.colorbar_kwargs == {}
    pm9 = PixelMap([[0]], plot_config={"colorbar_kwargs": {"key": "value"}})
    assert pm9.plotting.config.colorbar_kwargs == {"key": "value"}
    assert isinstance(pm3.plotting.config.colorbar_kwargs, frozendict.frozendict)


def test_update_plot_config():
    pp0 = PixelMapPlotConfig()
    assert isinstance(pp0, PixelMapPlotConfig)

    if sys.version_info >= (3, 13):
        pp1 = copy.replace(pp0, cmap="foo")
        assert pp1.cmap == "foo"
        assert pp1 == pp0.update(cmap="foo")

    pm0 = PixelMap([[0]])
    pm2 = PixelMap([[0]], plot_config=PixelMapPlotConfig(facecolor="white"))

    pm4 = pm0.update(plot_config=PixelMapPlotConfig(facecolor="white"))
    pm5 = pm0.update(plot_config={"facecolor": "white"})

    assert pm4.plotting.config != pm0.plotting.config
    assert pm4.plotting.config == pm2.plotting.config
    assert pm4.plotting.config == pm5.plotting.config

    pm6 = pm2.update(plot_config={"cmap": "inferno"})
    pm7 = pm2.update(plot_config=PixelMapPlotConfig(cmap="inferno"))
    pm8 = pm2.update(plot_config=pm2.plotting.config.update(cmap="inferno"))

    # when passing a dict as plot_config, plot_config is *updated*, not *replaced*
    # pm6 works the same as pm8 internally
    # when passing a PixelMapPlotConfig object, the entire object is replaced
    # pm7 should probably almost never be used
    assert pm6.plotting.config != pm7.plotting.config
    assert pm7.plotting.config.facecolor != "white"
    assert pm6.plotting.config == pm8.plotting.config

    pm9 = PixelMap([[0]], plot_config={"colorbar_kwargs": {"key": "value"}})

    # when updating, update colorbar kwargs instead of overwriting it
    pm10 = pm9.update(plot_config={"colorbar_kwargs": {"foo": "bar"}})
    assert pm10.plotting.config.colorbar_kwargs == {"key": "value", "foo": "bar"}

    # overriding existing value
    pm11 = pm9.update(plot_config={"colorbar_kwargs": {"key": "another value"}})
    assert pm11.plotting.config.colorbar_kwargs == {"key": "another value"}


def test_create_threshold_mask():
    pm = PixelMap([[-2, -1, 0, 1, 2]])

    mask = pm.create_mask_from_threshold(1)
    assert np.array_equal(mask.mask, [[np.nan, np.nan, np.nan, 1.0, 1.0]], equal_nan=True)

    mask = pm.create_mask_from_threshold(1, comparator=np.greater)
    assert np.array_equal(mask.mask, [[np.nan, np.nan, np.nan, np.nan, 1.0]], equal_nan=True)

    mask = pm.create_mask_from_threshold(1, use_magnitude=True)
    assert np.array_equal(mask.mask, [[1.0, 1.0, np.nan, 1.0, 1.0]], equal_nan=True)

    mask = pm.create_mask_from_threshold(1, comparator=np.less)
    assert np.array_equal(mask.mask, [[1.0, 1.0, 1.0, np.nan, np.nan]], equal_nan=True)


def test_convert():
    pm0 = PixelMap([[0]])
    pm1 = pm0.convert_to(PixelMap)
    assert pm0 == pm1

    pm2 = pm0.convert_to(PerfusionMap)
    assert pm0 != pm2
    assert isinstance(pm2, PerfusionMap)

    with pytest.raises(TypeError, match=r"`target_type` must be \(a subclass of\) PixelMap"):
        _ = pm0.convert_to(PixelMapPlotConfig)

    # TODO: test kwargs in `convert_to()`


def test_pixel_map():
    pm = PixelMap([[0]])

    assert pm._plot_config.cmap == "viridis"
    assert pm._plot_config.norm == "linear"

    assert isinstance(pm._plot_config.colorbar_kwargs, frozendict.frozendict)
    assert pm._plot_config.colorbar_kwargs == {}


def test_tiv_map():
    pm = TIVMap([[0]])

    assert isinstance(pm.plotting.config.cmap, Colormap)
    assert pm.plotting.config.cmap.name == "Blues_r"

    assert isinstance(pm.plotting.config.norm, Normalize)
    assert pm.plotting.config.norm.vmin == 0
    assert pm.plotting.config.norm.vmax is None

    assert isinstance(pm.plotting.config.colorbar_kwargs, frozendict.frozendict)
    assert pm.plotting.config.colorbar_kwargs["label"] == "TIV (a.u.)"


def test_imshow():
    pm = TIVMap(np.reshape(np.arange(100), (10, 10)))
    cm = pm.plotting.imshow()
    assert cm.axes == plt.gca()
    assert cm.cmap == pm.plotting.config.cmap
    axes = cm.figure.get_axes()
    assert len(axes) == 2  # colorbar is also an axes

    assert cm.axes in axes
    colorbar_axes = next(iter(set(axes) - {cm.axes}))

    assert colorbar_axes.get_ylabel() == pm.plotting.config.colorbar_kwargs["label"]


def test_normalize_values():
    pm0 = PixelMap(np.random.default_rng().random((10, 10), dtype=float))
    pm1 = pm0.normalize(mode="zero-based")

    assert pm1.values.min() == 0.0
    assert pm1.values.max() == 1.0

    pm2 = pm0.normalize(mode="symmetric")
    assert np.abs(pm2.values).max() == 1.0
    assert np.abs(pm2.values).min() >= 0.0

    pm3 = pm0 - 0.5
    pm4 = pm3.normalize(mode="symmetric")
    assert -1.0 <= pm4.values.min() <= 0.0
    assert 0.0 <= pm4.values.max() <= 1.0
    assert np.abs(pm4.values).max() == 1.0

    pm5 = pm3.normalize(mode="maximum")
    assert pm5.values.max() == 1.0
    assert pm5.values.min() >= -1.0 / pm3.values.max()

    pm6 = PixelMap(np.linspace(-10, 2, 100).reshape((10, 10)))
    pm7 = pm6.normalize(mode="maximum")
    assert pm7.values.max() == 1.0
    assert pm7.values.min() == -5.0

    pm8 = pm6.normalize(mode="reference", reference=4)
    assert pm8.values.max() == 0.5
    assert pm8.values.min() == -2.5

    pm9 = pm6.normalize(mode="reference", reference=0.1)
    assert pm9.values.max() == 20
    assert pm9.values.min() == -100


def test_initialize_with_nan():
    with pytest.warns(UserWarning, match="PixelMap initialized with all NaN values."):
        _ = PixelMap([[np.nan, np.nan], [np.nan, np.nan]])

    with warnings.catch_warnings(record=True) as w:
        _ = PixelMap([[np.nan, np.nan], [np.nan, np.nan]], suppress_all_nan_warning=True)  # does not warn
        assert len(w) == 0


def test_normalize_values_errors():
    pm = PixelMap(np.random.default_rng().random((10, 10), dtype=float))

    with pytest.raises(ValueError, match="Unknown normalization mode"):
        _ = pm.normalize(mode="non-existing")

    with pytest.raises(ValueError, match="`reference` can only be used with"):
        _ = pm.normalize(mode="maximum", reference=1)

    with pytest.raises(ValueError, match="`reference` must be provided when `mode='reference'`"):
        _ = pm.normalize(mode="reference")

    with pytest.raises(TypeError, match="`reference` must be a number"):
        _ = pm.normalize(mode="reference", reference="foo")

    with pytest.raises(TypeError, match="`reference` must be a number"):
        _ = pm.normalize(mode="reference", reference=[1, 2, 3])

    with pytest.raises(TypeError, match="`reference` must be a number"):
        _ = pm.normalize(mode="reference", reference=np.array([1, 2, 3]))

    with pytest.raises(ZeroDivisionError, match="Normalization by zero is not allowed"):
        _ = pm.normalize(mode="reference", reference=0)

    with pytest.raises(ZeroDivisionError, match="Normalization by zero is not allowed"):
        _ = PixelMap([[0]]).normalize()

    with pytest.raises(ValueError, match="Normalization by NaN is not allowed"):
        _ = pm.normalize(mode="reference", reference=np.nan)

    with pytest.raises(ValueError, match="Normalization by NaN is not allowed"):
        _ = PixelMap([[np.nan]]).normalize()

    with pytest.warns(UserWarning, match="Normalization by a negative number"):
        _ = pm.normalize(mode="reference", reference=-1)


def test_plot_extend():
    cmap = plt.colormaps["Greens"]

    # over and under are min and max values of colormap
    assert np.array_equal(cmap.get_under(), cmap(0.0))
    assert np.array_equal(cmap.get_over(), cmap(1.0))

    # colorbar has no extend
    cm0 = PixelMap([[0]], plot_config={"cmap": cmap}).plotting.imshow()
    assert cm0.colorbar.extend == "neither"

    # with under set to a different color, colorbar has min extend
    cmap.set_under("orange")
    cm1 = PixelMap([[0]], plot_config={"cmap": cmap}).plotting.imshow()
    assert cm1.colorbar.extend == "min"

    # with under and over set to a different color, colorbar has both extend
    cmap.set_over("red")
    cm2 = PixelMap([[0]], plot_config={"cmap": cmap}).plotting.imshow()
    assert cm2.colorbar.extend == "both"

    # with over set to a different color, colorbar has max extend
    cmap = plt.colormaps["Greens"]
    cmap.set_over("red")
    cm3 = PixelMap([[0]], plot_config={"cmap": cmap}).plotting.imshow()
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
    assert isinstance(pm.plotting.config.norm, CenteredNorm)

    cm = pm.plotting.imshow()
    assert cm.colorbar.norm.vcenter == 0
    assert cm.colorbar.norm.vmin == -10
    assert cm.colorbar.norm.vmax == 10


def test_add():
    pm1 = PerfusionMap([[0, 1, 2, 3]], plot_config={"absolute": True})
    pm1_add = pm1 + 3
    assert np.array_equal(pm1_add.values, np.array([[3, 4, 5, 6]]))
    assert isinstance(pm1_add, PerfusionMap)
    assert pm1_add.plotting.config == pm1.plotting.config

    add_pm1 = 3 + pm1
    assert np.array_equal(pm1_add.values, add_pm1.values)
    assert isinstance(add_pm1, PerfusionMap)
    assert add_pm1.plotting.config == pm1.plotting.config

    pm2 = TIVMap([[1, 2, 3, 4]])
    pm1_add_pm2 = pm1 + pm2
    assert isinstance(pm1_add_pm2, PixelMap)
    assert np.array_equal(pm1_add_pm2.values, [[1, 3, 5, 7]])

    pm3 = PixelMap([[1, 2, 3]])
    with pytest.raises(ValueError, match=r"Shape of PixelMaps \(self: \(1, 4\), other: \(1, 3\)\) do not match."):
        _ = pm1 + pm3


def test_sub():
    pm1 = PerfusionMap([[0, 1, 2, 3]], plot_config={"absolute": True})
    pm1_sub = pm1 - 3
    assert np.array_equal(pm1_sub.values, np.array([[-3, -2, -1, 0]]))
    assert isinstance(pm1_sub, PerfusionMap)
    assert pm1_sub.plotting.config == pm1.plotting.config

    sub_pm1 = 3 - pm1
    assert np.array_equal(sub_pm1.values, [[3, 2, 1, 0]])
    assert isinstance(sub_pm1, PerfusionMap)
    assert sub_pm1.plotting.config == pm1.plotting.config

    pm2 = TIVMap([[1, 2, 3, 4]])
    pm1_sub_pm2 = pm1 - pm2
    assert isinstance(pm1_sub_pm2, DifferenceMap)
    assert np.array_equal(pm1_sub_pm2.values, [[-1, -1, -1, -1]])
    pm2_sub_pm1 = pm2 - pm1
    assert isinstance(pm2_sub_pm1, DifferenceMap)
    assert np.array_equal(pm2_sub_pm1.values, [[1, 1, 1, 1]])


def test_mul():
    pm1 = PerfusionMap([[0, 1, 2, 3]], plot_config={"absolute": True})
    pm1_mul = pm1 * 3
    assert np.array_equal(pm1_mul.values, np.array([[0, 3, 6, 9]]))
    assert isinstance(pm1_mul, PerfusionMap)
    assert pm1_mul.plotting.config == pm1.plotting.config

    mul_pm1 = 3 * pm1
    assert np.array_equal(pm1_mul.values, mul_pm1.values)
    assert isinstance(mul_pm1, PerfusionMap)
    assert mul_pm1.plotting.config == pm1.plotting.config

    pm2 = TIVMap([[1, 2, 3, 4]])
    pm1_mul_pm2 = pm1 * pm2
    assert isinstance(pm1_mul_pm2, PixelMap)
    assert np.array_equal(pm1_mul_pm2.values, [[0, 2, 6, 12]])

    pm2_mul_pm1 = pm2 * pm1
    assert isinstance(pm2_mul_pm1, PixelMap)
    assert np.array_equal(pm1_mul_pm2.values, pm2_mul_pm1.values)


def test_div():
    pm1 = PerfusionMap([[0, 1, 2, 3]], plot_config={"absolute": True})
    pm1_div = pm1 / 3
    assert np.array_equal(pm1_div.values, np.array([[0, 1 / 3, 2 / 3, 1]]))
    assert isinstance(pm1_div, PerfusionMap)
    assert pm1_div.plotting.config == pm1.plotting.config

    with pytest.warns(UserWarning, match="Dividing by 0 will result in `np.nan`"):
        div_pm1 = 3 / pm1
    assert np.array_equal(div_pm1.values, [[np.nan, 3, 3 / 2, 1]], equal_nan=True)
    assert isinstance(div_pm1, PerfusionMap)
    assert div_pm1.plotting.config == pm1.plotting.config

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
    mean_pm1_pm2 = ODCLMap.from_aggregate([pm1, pm2], np.mean)

    assert np.array_equal(mean_pm1_pm2.values, [[0.5, 1.5, 2.5, 3.5]])
    assert isinstance(mean_pm1_pm2, ODCLMap)

    pm3 = PendelluftMap([[np.nan, 3, 4, np.nan]])
    pm4 = DifferenceMap([[np.nan, np.nan, 5, 6]])
    mean_pm3_pm4 = SignedPendelluftMap.from_aggregate([pm3, pm4], np.nanmean)
    assert isinstance(mean_pm3_pm4, SignedPendelluftMap)
    assert np.array_equal(mean_pm3_pm4.values, [[np.nan, 3, 4.5, 6]], equal_nan=True)

    mean_all = PixelMap.from_aggregate([pm1, pm2, pm3, pm4], np.nanmean, label="foo", plot_config={"colorbar": False})
    assert isinstance(mean_all, PixelMap)
    assert np.array_equal(mean_all.values, np.array([[0.5, 2, 3.5, 13 / 3]]))
    assert mean_all.label == "foo"
    assert mean_all._plot_config.colorbar is False

    array1 = [[0, 1, 2, 3]]
    array2 = [[2, 3, 4, 5]]
    mean_array1_array2 = TIVMap.from_aggregate([array1, array2], np.mean)
    assert isinstance(mean_array1_array2, TIVMap)
    assert np.array_equal(mean_array1_array2.values, [[1, 2, 3, 4]])

    array3 = [[[4, 5, 6, 7]]]
    with pytest.raises(ValueError, match="should have 2 dimensions, not 3"):
        _ = PixelMap.from_aggregate([array3], np.mean)

    array4 = [[4, 5, 6, 7], [5, 6, 7, 8]]
    with pytest.raises(ValueError, match="all input arrays must have the same shape"):
        _ = PixelMap.from_aggregate([array1, array4], np.mean)


def test_replace_defaults():
    """Test that replacing PixelMapPlotParameters defaults works as expected."""
    pm0 = PixelMap([[0]])
    assert pm0.plotting.config.cmap == "viridis"

    _PLOT_CONFIG_REGISTRY[PixelMap] = PixelMapPlotConfig(cmap="plasma")

    pm1 = PixelMap([[0]])
    assert pm0.plotting.config.cmap == "viridis"
    assert pm1.plotting.config.cmap == "plasma"

    pm2 = TIVMap([[0]])
    assert isinstance(pm2.plotting.config.cmap, Colormap)
    assert pm2.plotting.config.cmap.name == "Blues_r"

    _PLOT_CONFIG_REGISTRY[TIVMap] = _PLOT_CONFIG_REGISTRY[TIVMap].update(cmap="Greens")

    pm3 = TIVMap([[0]])
    assert pm3.plotting.config.cmap == "Greens"
    assert pm3.plotting.config.colorbar

    _PLOT_CONFIG_REGISTRY[TIVMap] = _PLOT_CONFIG_REGISTRY[TIVMap].update(colorbar=False)

    pm4 = TIVMap([[0]])
    assert pm4.plotting.config.cmap == "Greens"
    assert not pm4.plotting.config.colorbar

    reset_plot_config()
    pm0 = PixelMap([[0]])
    assert pm0.plotting.config.cmap == "viridis"


def test_set_pixelmap_plot_parameters():
    """Test that set_pixelmap_plot_parameters works as expected."""
    pm0 = PixelMap([[0]])
    assert pm0.plotting.config.cmap == "viridis"

    set_plot_config_parameters(PixelMap, cmap="plasma")

    pm1 = PixelMap([[0]])
    assert pm1.plotting.config.cmap == "plasma"

    pm2 = TIVMap([[0]])
    assert isinstance(pm2.plotting.config.cmap, Colormap)
    assert pm2.plotting.config.cmap.name == "Blues_r"

    set_plot_config_parameters(TIVMap, cmap="Greens")

    pm3 = TIVMap([[0]])
    assert pm3.plotting.config.cmap == "Greens"

    set_plot_config_parameters(cmap="Reds", colorbar=False)
    assert all(
        cls([[0]]).plotting.config.cmap == "Reds" and not cls([[0]]).plotting.config.colorbar
        for cls in _PLOT_CONFIG_REGISTRY
        if not isinstance(cls, str)
    )

    reset_plot_config(PixelMap)

    pm4 = PixelMap([[0]])
    pm5 = TIVMap([[0]])
    assert pm4.plotting.config.cmap == "viridis"
    assert pm5.plotting.config.cmap == "Reds"


def test_dtype():
    pm1 = IntegerMap([[0, 1], [2, 3]])
    assert pm1.values.dtype == np.int_

    pm2 = IntegerMap([[0.0, 1.0], [2.0, 3.0]])
    assert pm2.values.dtype == np.int_

    pm3 = PixelMap([[0, 1], [2, 3]])
    assert pm3.values.dtype == np.float64, "integers should be convertable to float64"

    pm3 = PixelMap([[0 + 0j, 1.1], [2, 3]])
    assert pm3.values.dtype == np.float64, "complex numbers without imaginary part should be convertable to float64"

    with pytest.raises(TypeError, match="Values must be convertible to"):
        _ = IntegerMap([[0.5, 1], [2, 3]])

    with pytest.raises(TypeError, match="Values must be convertible to"):
        _ = PixelMap([["a", 1], [2, 3]])
