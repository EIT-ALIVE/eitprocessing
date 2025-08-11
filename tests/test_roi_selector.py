import numpy as np
import pytest
from scipy.ndimage import generate_binary_structure

from eitprocessing.roi import PixelMask
from eitprocessing.roi.roi_selector import ROISelector


def make_pixel_mask(array: np.ndarray):
    """Helper to wrap numpy array into PixelMask with NaNs for False."""
    return PixelMask(array, keep_zeros=True)


def test_basic_region_selection():
    arr = np.full((5, 5), np.nan)
    arr[1:4, 1:4] = True
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_region_size=5, structure="4-connectivity")
    result = selector.select_regions(mask)
    expected_mask = np.full(arr.shape, np.nan)
    expected_mask[1:4, 1:4] = 1
    np.testing.assert_array_equal(result.mask, expected_mask)


def test_region_smaller_than_threshold_is_excluded():
    arr = np.full((5, 5), np.nan)
    arr[0, 0:2] = True
    arr[2:4, 2:5] = True
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_region_size=5, structure="4-connectivity")
    result = selector.select_regions(mask)
    expected = np.zeros_like(arr, dtype=float)
    expected[2:4, 2:5] = 1
    expected[expected == 0] = np.nan
    np.testing.assert_array_equal(result.mask, expected)


def test_no_regions_above_threshold_warns_and_returns_empty():
    arr = np.full((4, 4), np.nan)
    arr[0, 0] = True
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_region_size=5, structure="4-connectivity")
    with pytest.warns(UserWarning, match="No regions found above min_pixels threshold."):
        result = selector.select_regions(mask)
    assert isinstance(result, PixelMask)
    assert np.all(np.isnan(result.mask))


def test_custom_connectivity():
    arr = np.full((3, 3), np.nan)
    arr[0, 0] = True
    arr[1, 1] = True
    mask = make_pixel_mask(arr)
    # Default (4-connectivity) — should warn and return empty
    selector_default = ROISelector(min_region_size=2, structure="4-connectivity")
    with pytest.warns(UserWarning, match="No regions found above min_pixels threshold."):
        result_default = selector_default.select_regions(mask)
    assert np.all(np.isnan(result_default.mask))
    # 8-connectivity — diagonal pixels connected
    selector_diag = ROISelector(min_region_size=2, structure="8-connectivity")
    result_diag = selector_diag.select_regions(mask)
    assert not np.all(np.isnan(result_diag.mask))
    # Custom structure
    custom_structure = generate_binary_structure(2, 2)
    selector_custom = ROISelector(min_region_size=2, structure=custom_structure)
    result_custom = selector_custom.select_regions(mask)
    assert not np.all(np.isnan(result_custom.mask))


def test_empty_mask_returns_empty():
    arr = np.full((4, 4), np.nan)
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_region_size=1, structure="4-connectivity")
    result = selector.select_regions(mask)
    assert np.all(np.isnan(result.mask))


def test_all_nan_mask_returns_empty():
    arr = np.full((4, 4), np.nan)
    mask = PixelMask(arr)
    selector = ROISelector(min_region_size=1, structure="4-connectivity")
    result = selector.select_regions(mask)
    assert np.all(np.isnan(result.mask))


def test_zeros_are_regions_nans_are_excluded():
    arr = np.array([[0, 1, np.nan], [0, 0.5, np.nan], [np.nan, np.nan, 0]], dtype=float)
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_region_size=3, structure="4-connectivity")
    result_mask = selector.select_regions(mask).mask
    result_binary = ~np.isnan(result_mask)
    expected_binary = np.zeros_like(arr, dtype=bool)
    expected_binary[0:2, 0:2] = True
    np.testing.assert_array_equal(result_binary, expected_binary)


def test_multiple_large_regions_are_all_included():
    arr = np.full((6, 6), np.nan)
    arr[0:2, 0:2] = True
    arr[4:6, 4:6] = True
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_region_size=4, structure="4-connectivity")
    result = selector.select_regions(mask)
    assert np.sum(~np.isnan(result.mask)) == 8


def test_all_true_mask_returns_full_mask():
    arr = np.ones((3, 3), dtype=bool)
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_region_size=1, structure="4-connectivity")
    result = selector.select_regions(mask)
    assert np.all(result.mask == 1)


def test_edge_connected_region():
    arr = np.full((5, 5), np.nan)
    arr[0, :] = True
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_region_size=5, structure="4-connectivity")
    result = selector.select_regions(mask)
    assert np.sum(~np.isnan(result.mask)) == 5


def test_min_pixels_threshold_variation():
    arr = np.full((5, 5), np.nan)
    arr[1:3, 1:3] = True
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_region_size=4, structure="4-connectivity")
    result = selector.select_regions(mask)
    assert np.sum(~np.isnan(result.mask)) == 4
    selector2 = ROISelector(min_region_size=5, structure="4-connectivity")
    result2 = selector2.select_regions(mask)
    assert np.all(np.isnan(result2.mask))


def test_touching_regions_are_separated():
    arr = np.full((5, 5), np.nan)
    arr[1:3, 1:3] = True
    arr[3, 1:3] = True
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_region_size=2, structure="4-connectivity")
    result = selector.select_regions(mask)
    assert np.sum(~np.isnan(result.mask)) == 6
    selector8 = ROISelector(min_region_size=6, structure="8-connectivity")
    result8 = selector8.select_regions(mask)
    assert np.sum(~np.isnan(result8.mask)) == 6


def test_structure_input_variants():
    """Test all possible structure inputs: None, string, array."""
    arr = np.full((4, 4), np.nan)
    arr[1:3, 1:3] = True
    mask = make_pixel_mask(arr)
    # None (should default to 4-connectivity)
    selector_none = ROISelector(min_region_size=4, structure=None)
    result_none = selector_none.select_regions(mask)
    assert np.sum(~np.isnan(result_none.mask)) == 4
    # String
    selector_str = ROISelector(min_region_size=4, structure="4-connectivity")
    result_str = selector_str.select_regions(mask)
    assert np.sum(~np.isnan(result_str.mask)) == 4
    # Array
    structure_arr = generate_binary_structure(2, 1)
    selector_arr = ROISelector(min_region_size=4, structure=structure_arr)
    result_arr = selector_arr.select_regions(mask)
    assert np.sum(~np.isnan(result_arr.mask)) == 4
