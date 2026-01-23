import warnings

import numpy as np
import pytest

from eitprocessing.datahandling.pixelmap import PixelMap
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.roi import PixelMask, get_geometric_mask


def test_pixelmask_init_with_boolean_array():
    values = np.random.default_rng().random((10, 10)) > 0.5
    assert np.all((values == True) | (values == False))  # noqa: E712

    mask = PixelMask(values)

    assert np.all((np.isnan(mask.mask)) | (mask.mask == 1.0))
    assert np.array_equal(np.isnan(mask.mask), ~values)
    assert np.array_equal(mask.mask == 1.0, values)


def test_pixelmask_init_with_integer_array():
    values = np.random.default_rng().integers(0, 2, (10, 10))
    mask = PixelMask(values, suppress_zero_conversion_warning=True)

    assert np.all((np.isnan(mask.mask)) | (mask.mask == 1.0))
    assert np.array_equal(np.isnan(mask.mask), values == 0)
    assert np.array_equal(mask.mask == 1.0, values == 1)


def test_pixelmask_init_with_float_array_non_weighted():
    values = np.random.default_rng().random((10, 10)).round()
    assert np.all((values == 0.0) | (values == 1.0))

    mask = PixelMask(values, suppress_zero_conversion_warning=True)

    assert np.all((np.isnan(mask.mask)) | (mask.mask == 1.0))
    assert np.array_equal(np.isnan(mask.mask), values == 0.0)
    assert np.array_equal(mask.mask == 1.0, values == 1.0)


def test_pixelmask_init_with_float_array_weighted():
    values = np.random.default_rng().random((10, 10))
    values[values < 0.5] = 0.0

    mask = PixelMask(values, suppress_zero_conversion_warning=True)

    assert np.array_equal(np.isnan(mask.mask), values < 0.5)  # NaN for values < 0.5
    assert np.array_equal(~np.isnan(mask.mask), values >= 0.5)  # non-NaN for values >= 0.5
    assert np.array_equal(mask.mask[~np.isnan(mask.mask)], values[~np.isnan(mask.mask)])  # non-NaN values are unaltered


def test_pixelmask_init_with_list():
    values = [[1, 0, 1], [0, 1, 0]]
    mask = PixelMask(values, suppress_zero_conversion_warning=True)

    assert np.array_equal(mask.mask, np.array([[1.0, np.nan, 1.0], [np.nan, 1.0, np.nan]]), equal_nan=True)


def test_pixelmask_init_keep_zeros_true():
    values = np.random.default_rng().random((10, 10))
    values[values < 0.5] = 0.0

    mask = PixelMask(values, keep_zeros=True)

    assert np.array_equal(mask.mask == 0.0, values < 0.5)  # NaN for values < 0.5
    assert np.array_equal(mask.mask[mask.mask >= 0.5], values[mask.mask >= 0.5])  # non-NaN values are unaltered


def test_pixelmask_warns_when_keep_zeros_false():
    values = np.random.default_rng().random((10, 10))
    values[values < 0.5] = 0.0

    with pytest.warns(UserWarning, match="Mask contains 0 values, which will be converted to NaN"):
        _ = PixelMask(values)


def test_pixelmask_does_not_warn_when_boolean_array_has_zeros():
    values = np.random.default_rng().random((10, 10))
    values = values > 0.5  # boolean array

    with warnings.catch_warnings(record=True) as w:
        _ = PixelMask(values)  # as boolean array
        assert len(w) == 0

    with pytest.warns(UserWarning, match="Mask contains 0 values, which will be converted to NaN"):
        _ = PixelMask(values.astype(int))  # the same values, but as integer array


def test_pixelmask_init_invalid_dtype():
    with pytest.raises(ValueError, match="could not convert string to float"):
        PixelMask([["string"]])

    with pytest.raises(TypeError, match="float\\(\\) argument must be a string or a (real )?number"):
        PixelMask([[lambda x: x]])

    with pytest.raises(TypeError, match="float\\(\\) argument must be a string or a real number, not 'complex'"):
        PixelMask([[complex(1, 2)]])

    # is converted to float
    PixelMask(np.array(np.random.default_rng().random((10, 10)), dtype="object"))


def test_pixelmask_init_values_outside_range():
    with pytest.raises(ValueError, match="One or more mask values fall outside the range 0 to 1"):
        _ = PixelMask(np.array([[1.5, 0.2], [0.5, 0.8]]))

    with pytest.raises(ValueError, match="One or more mask values fall outside the range 0 to 1"):
        _ = PixelMask(np.array([[0.5, -0.2], [0.5, 0.8]]))


def test_pixelmask_init_suppress_value_range_warning():
    _ = PixelMask(np.array([[1.5, 0.2], [0.5, 0.8]]), suppress_value_range_error=True)
    _ = PixelMask(np.array([[0.5, -0.2], [0.5, 0.8]]), suppress_value_range_error=True)


def test_pixelmask_init_dimension_mismatch():
    with pytest.raises(ValueError, match="Mask should be a 2D array, not 3D"):
        _ = PixelMask(np.ones((3, 3, 3)))

    with pytest.raises(ValueError, match="Mask should be a 2D array, not 1D"):
        _ = PixelMask(np.ones((3,)))


def test_pixelmask_is_weighted_true():
    pm = PixelMask(np.random.default_rng().random((10, 10)))
    assert pm.is_weighted


def test_pixelmask_is_weighted_false():
    pm = PixelMask(np.round(np.random.default_rng().random((10, 10))), suppress_zero_conversion_warning=True)
    assert not pm.is_weighted


def test_pixelmask_apply_numpy_array():
    pm = PixelMask([[0, 1], [1, 0]], suppress_zero_conversion_warning=True)
    data = np.array([[1, 2], [3, 4]])
    result = pm.apply(data)
    assert np.array_equal(result, np.array([[np.nan, 2], [3, np.nan]]), equal_nan=True)

    pm = PixelMask([[0.1, 0.9], [0.2, 0.5]])
    data = np.array([[1, 2], [3, 4]])
    result = pm.apply(data)
    assert np.allclose(result, np.array([[0.1, 1.8], [0.6, 2.0]]))


def test_pixelmask_apply_numpy_array_higher_dimensions():
    pm = PixelMask([[0, 1], [1, 0]], suppress_zero_conversion_warning=True)
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = pm.apply(data)
    assert np.array_equal(result, np.array([[[np.nan, 2], [3, np.nan]], [[np.nan, 6], [7, np.nan]]]), equal_nan=True)


def test_pixelmask_apply_eitdata(draeger_20hz_healthy_volunteer: Sequence):
    eit_data = draeger_20hz_healthy_volunteer.eit_data["raw"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        mask = PixelMask(np.full((32, 32), np.nan), suppress_all_nan_warning=True)
    masked_eit_data = mask.apply(eit_data)

    assert masked_eit_data.pixel_impedance.shape == eit_data.pixel_impedance.shape
    assert np.all(np.isnan(masked_eit_data.pixel_impedance))

    mask_values = np.full((32, 32), np.nan)
    mask_values[10:23, 10:23] = 1.0  # let center pixels pass
    mask = PixelMask(mask_values)
    masked_eit_data = mask.apply(eit_data)

    assert np.array_equal(masked_eit_data.pixel_impedance[:, 10:23, 10:23], eit_data.pixel_impedance[:, 10:23, 10:23])
    assert np.all(np.isnan(masked_eit_data.pixel_impedance[:, :10, :]))
    assert np.all(np.isnan(masked_eit_data.pixel_impedance[:, 23:, :]))
    assert np.all(np.isnan(masked_eit_data.pixel_impedance[:, :, :10]))
    assert np.all(np.isnan(masked_eit_data.pixel_impedance[:, :, 23:]))


def test_pixelmask_apply_pixelmap():
    pmap = PixelMap(np.random.default_rng().random((10, 10)))
    mask = PixelMask(np.random.default_rng().random((10, 10)) > 0.5)  # Mask with some values > 0.5
    masked_pmap = mask.apply(pmap)

    assert masked_pmap.shape == pmap.shape
    assert np.all(np.isnan(masked_pmap.values[np.isnan(mask.mask)]))
    assert np.array_equal(pmap.values[~np.isnan(mask.mask)], masked_pmap.values[~np.isnan(mask.mask)])


def test_pixelmask_apply_invalid_type():
    with pytest.raises(TypeError, match="Data should be an array, or EITData or PixelMap object, not <class 'str'>"):
        PixelMask([[1]]).apply("invalid type")

    with pytest.raises(TypeError, match="Data should be an array, or EITData or PixelMap object, not <class 'list'>"):
        PixelMask([[1]]).apply([[1]])


def test_pixelmask_apply_dimension_mismatch():
    pm = PixelMask(np.random.default_rng().random((10, 10)))
    data = np.random.default_rng().random((10, 10, 10, 9))

    with pytest.raises(ValueError, match="Data shape .* does not match Mask shape .*"):
        pm.apply(data)


def test_pixelmask_multiply_masks():
    pm1 = PixelMask([[0, 0.1], [0.2, 0.3]], suppress_zero_conversion_warning=True)
    pm2 = PixelMask([[0.1, 0.2], [0.3, 0.4]], suppress_zero_conversion_warning=True)
    pm3 = pm1 * pm2
    assert np.allclose(pm3.mask, np.array([[np.nan, 0.02], [0.06, 0.12]]), equal_nan=True)


def test_pixelmask_add_masks():
    pm1 = PixelMask([[0, 0, 1, 1], [0.2, 0.3, 0, 0]], suppress_zero_conversion_warning=True)
    pm2 = PixelMask([[1, 0, 0, 1], [0, 0.5, 1, 0]], suppress_zero_conversion_warning=True)
    pm3 = pm1 + pm2
    assert np.array_equal(pm3.mask, np.array([[1, np.nan, 1, 1], [0.2, 0.8, 1, np.nan]]), equal_nan=True)


def test_pixelmask_subtract():
    pm1 = PixelMask([[0, 0, 1, 1], [0.2, 0.3, 0.6, 0.5]], suppress_zero_conversion_warning=True)
    pm2 = PixelMask([[1, 0, 0, 1], [0.1, 0.5, 0.6, 0]], suppress_zero_conversion_warning=True)
    pm3 = pm1 - pm2
    assert np.array_equal(pm3.mask, np.array([[np.nan, np.nan, 1, np.nan], [0.1, np.nan, np.nan, 0.5]]), equal_nan=True)


@pytest.mark.parametrize("shape", [(32, 1), (64, 1), (16, 1), (100, 1), (4, 1)])
def test_predefined_global_mask(shape: tuple[int, int]):
    global_mask = get_geometric_mask("global", shape)
    assert global_mask.label == "global"
    assert global_mask.shape == shape
    assert np.all(global_mask.mask == 1.0)


@pytest.mark.parametrize("shape", [(32, 1), (64, 1), (16, 1), (100, 1), (4, 1)])
def test_predefined_layer_masks(shape: tuple[int, int]):
    quarter_height = shape[0] // 4
    first = slice(None, quarter_height)
    second = slice(quarter_height, 2 * quarter_height)
    third = slice(2 * quarter_height, 3 * quarter_height)
    fourth = slice(3 * quarter_height, None)
    layer1_mask = get_geometric_mask("layer 1", shape)

    assert layer1_mask.label == "layer 1"
    assert layer1_mask == get_geometric_mask("L1", shape)
    assert get_geometric_mask("L1").label == "layer 1"
    assert layer1_mask.shape == shape

    assert np.all(layer1_mask.mask[first, :] == 1.0)
    assert np.all(np.isnan(layer1_mask.mask[second, :]))
    assert np.all(np.isnan(layer1_mask.mask[third, :]))
    assert np.all(np.isnan(layer1_mask.mask[fourth, :]))

    layer2_mask = get_geometric_mask("layer 2", shape)
    assert layer2_mask.label == "layer 2"
    assert np.all(np.isnan(layer2_mask.mask[first, :]))
    assert np.all(layer2_mask.mask[second, :] == 1.0)
    assert np.all(np.isnan(layer2_mask.mask[third, :]))
    assert np.all(np.isnan(layer2_mask.mask[fourth, :]))

    layer3_mask = get_geometric_mask("layer 3", shape)
    assert layer3_mask.label == "layer 3"
    assert np.all(np.isnan(layer3_mask.mask[first, :]))
    assert np.all(np.isnan(layer3_mask.mask[second, :]))
    assert np.all(layer3_mask.mask[third, :] == 1.0)
    assert np.all(np.isnan(layer3_mask.mask[fourth, :]))

    layer4_mask = get_geometric_mask("layer 4", shape)
    assert layer4_mask.label == "layer 4"
    assert np.all(np.isnan(layer4_mask.mask[first, :]))
    assert np.all(np.isnan(layer4_mask.mask[second, :]))
    assert np.all(np.isnan(layer4_mask.mask[third, :]))
    assert np.all(layer4_mask.mask[fourth, :] == 1.0)

    assert np.array_equal((layer1_mask + layer2_mask + layer3_mask + layer4_mask).mask, np.ones(shape))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mask contains only NaN values. This will create in all-NaN results when applied",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        assert np.array_equal(
            (layer1_mask * layer2_mask * layer3_mask * layer4_mask).mask, np.full(shape, np.nan), equal_nan=True
        )


def test_predefined_ventral_dorsal_masks():
    ventral_mask = get_geometric_mask("ventral", (32, 32))
    assert ventral_mask.label == "ventral"
    assert ventral_mask.shape == (32, 32)
    assert ventral_mask == get_geometric_mask("V")
    assert get_geometric_mask("V").label == "ventral"

    dorsal_mask = get_geometric_mask("dorsal", (32, 32))
    assert dorsal_mask.label == "dorsal"

    assert np.all(ventral_mask.mask[:16, :] == 1.0)
    assert np.all(np.isnan(ventral_mask.mask[16:, :]))

    assert np.all(dorsal_mask.mask[16:, :] == 1.0)
    assert np.all(np.isnan(dorsal_mask.mask[:16, :]))

    assert np.array_equal(np.ones((32, 32)), (ventral_mask + dorsal_mask).mask)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mask contains only NaN values. This will create in all-NaN results when applied",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        assert np.array_equal(np.full((32, 32), np.nan), (ventral_mask * dorsal_mask).mask, equal_nan=True)


def test_predefined_left_right_masks():
    right_mask = get_geometric_mask("anatomical right", (32, 32))

    assert right_mask.label == "anatomical right"
    assert right_mask.shape == (32, 32)

    assert np.all(right_mask.mask[:, :16] == 1.0)
    assert np.all(np.isnan(right_mask.mask[:, 16:]))

    left_mask = get_geometric_mask("anatomical left", (32, 32))
    assert left_mask.label == "anatomical left"
    assert np.all(left_mask.mask[:, 16:] == 1.0)
    assert np.all(np.isnan(left_mask.mask[:, :16]))

    assert np.array_equal(np.ones((32, 32)), (right_mask + left_mask).mask)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mask contains only NaN values. This will create in all-NaN results when applied",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        assert np.array_equal(np.full((32, 32), np.nan), (right_mask * left_mask).mask, equal_nan=True)


def test_predefined_quadrant_masks():
    quadrant1_mask = get_geometric_mask("quadrant 1", (32, 32))

    assert quadrant1_mask.label == "quadrant 1"
    assert quadrant1_mask.shape == (32, 32)
    assert quadrant1_mask == get_geometric_mask("Q1")
    assert get_geometric_mask("Q1").label == "quadrant 1"

    assert np.all(quadrant1_mask.mask[:16, :16] == 1.0)
    assert np.all(np.isnan(quadrant1_mask.mask[16:, :]))
    assert np.all(np.isnan(quadrant1_mask.mask[:, 16:]))

    quadrant2_mask = get_geometric_mask("quadrant 2", (32, 32))
    assert np.all(quadrant2_mask.mask[:16, 16:] == 1.0)
    assert np.all(np.isnan(quadrant2_mask.mask[16:, :]))
    assert np.all(np.isnan(quadrant2_mask.mask[:, :16]))

    quadrant3_mask = get_geometric_mask("quadrant 3", (32, 32))
    assert np.all(quadrant3_mask.mask[16:, :16] == 1.0)
    assert np.all(np.isnan(quadrant3_mask.mask[:16, :]))
    assert np.all(np.isnan(quadrant3_mask.mask[:, 16:]))

    quadrant4_mask = get_geometric_mask("quadrant 4", (32, 32))
    assert np.all(quadrant4_mask.mask[16:, 16:] == 1.0)
    assert np.all(np.isnan(quadrant4_mask.mask[:16, :]))
    assert np.all(np.isnan(quadrant4_mask.mask[:, :16]))

    assert np.array_equal((quadrant1_mask + quadrant2_mask + quadrant3_mask + quadrant4_mask).mask, np.ones((32, 32)))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mask contains only NaN values. This will create in all-NaN results when applied",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        assert np.array_equal(
            (quadrant1_mask * quadrant2_mask * quadrant3_mask * quadrant4_mask).mask,
            np.full((32, 32), np.nan),
            equal_nan=True,
        )


def test_pixelmask_immutability():
    pm = PixelMask([[0, 1], [1, 0]], suppress_zero_conversion_warning=True)
    original_mask = pm.mask.copy()

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        pm.mask[0, 0] = 2  # Attempt to modify the mask should raise an error

    with pytest.raises(AttributeError, match="cannot assign to field 'mask'"):
        pm.mask = np.array([[1, 0], [0, 1]])

    assert np.array_equal(pm.mask, original_mask, equal_nan=True)  # Ensure the mask is unchanged


@pytest.mark.parametrize(
    ("mask_name", "shape"),
    [
        ("ventral", (33, 32)),
        ("dorsal", (33, 32)),
        ("layer 1", (18, 16)),
        ("layer 3", (15, 16)),
        ("quadrant 1", (16, 15)),
        ("quadrant 2", (15, 16)),
        ("quadrant 3", (15, 15)),
        ("anatomical right", (32, 33)),
        ("anatomical left", (32, 33)),
    ],
)
def test_get_geometric_mask_raises(mask_name: str, shape: tuple[int, int]):
    with pytest.raises(ValueError, match=r"Shape \(.*\) is not compatible with a .* mask."):
        _ = get_geometric_mask(mask_name, shape)


def test_plotting_works():
    _ = get_geometric_mask("layer 1", (32, 32)).plotting.imshow()
