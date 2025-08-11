import numpy as np
import pytest
from scipy.ndimage import generate_binary_structure

from eitprocessing.roi import PixelMask
from eitprocessing.roi.roi_selector import ROISelector


def make_pixel_mask(array: np.ndarray):
    """Helper to wrap numpy array into PixelMask with NaNs for False."""
    return PixelMask(array, keep_zeros=True)


def test_basic_region_selection():
    # Simple 5x5 mask with one large connected region
    arr = np.full((5, 5), np.nan)
    arr[1:4, 1:4] = True  # 3x3 region
    mask = make_pixel_mask(arr)

    selector = ROISelector(min_pixels=5)
    result = selector.select_regions(mask)

    # The combined mask should have the same shape and True in the same places
    expected_mask = np.full(arr.shape, np.nan)
    expected_mask[1:4, 1:4] = 1
    np.testing.assert_array_equal(result.mask, expected_mask)


def test_region_smaller_than_threshold_is_excluded():
    # One small region (2 pixels) and one large region (6 pixels)
    arr = np.full((5, 5), np.nan)
    arr[0, 0:2] = True  # small region
    arr[2:4, 2:5] = True  # large region
    mask = make_pixel_mask(arr)

    selector = ROISelector(min_pixels=5)
    result = selector.select_regions(mask)

    # Small region should be excluded
    expected = np.zeros_like(arr, dtype=float)
    expected[2:4, 2:5] = 1
    expected[expected == 0] = np.nan

    np.testing.assert_array_equal(result.mask, expected)


def test_no_regions_above_threshold_warns_and_returns_empty():
    # Mask has a single small region below threshold
    arr = np.full((4, 4), np.nan)
    arr[0, 0] = True  # 1 pixel only
    mask = make_pixel_mask(arr)

    selector = ROISelector(min_pixels=5)
    with pytest.warns(UserWarning, match="No regions found above min_pixels threshold."):
        result = selector.select_regions(mask)

    assert isinstance(result, PixelMask)
    # All NaN means no region kept
    assert np.all(np.isnan(result.mask))


def test_custom_connectivity():
    # Two diagonal pixels — only connected if using full connectivity
    arr = np.full((3, 3), np.nan)
    arr[0, 0] = True
    arr[1, 1] = True
    mask = make_pixel_mask(arr)

    # Default (4-connectivity) — should warn and return empty
    selector_default = ROISelector(min_pixels=2)
    with pytest.warns(UserWarning, match="No regions found above min_pixels threshold."):
        result_default = selector_default.select_regions(mask)
    assert np.all(np.isnan(result_default.mask))

    # 8-connectivity — diagonal pixels connected
    structure = generate_binary_structure(2, 2)  # 8-connected
    selector_diag = ROISelector(min_pixels=2, structure=structure)
    result_diag = selector_diag.select_regions(mask)
    assert not np.all(np.isnan(result_diag.mask))


def test_empty_mask_returns_empty():
    arr = np.full((4, 4), np.nan)
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_pixels=1)
    result = selector.select_regions(mask)
    assert np.all(np.isnan(result.mask))


def test_all_nan_mask_returns_empty():
    arr = np.full((4, 4), np.nan)
    mask = PixelMask(arr)
    selector = ROISelector(min_pixels=1)
    result = selector.select_regions(mask)
    assert np.all(np.isnan(result.mask))


def test_zeros_are_regions_nans_are_excluded():
    # Array with a mix of 0, 1, and NaNs (all within allowed range)
    arr = np.array([[0, 1, np.nan], [0, 0.5, np.nan], [np.nan, np.nan, 0]], dtype=float)

    # PixelMask should treat all non-NaN (including zeros) as valid region pixels
    mask = make_pixel_mask(arr)

    selector = ROISelector(min_pixels=3)  # region size threshold
    result_mask = selector.select_regions(mask).mask
    result_binary = ~np.isnan(result_mask)

    # Expected binary representation
    expected_binary = np.zeros_like(arr, dtype=bool)
    expected_binary[0:2, 0:2] = True  # only the top-left block survives

    np.testing.assert_array_equal(result_binary, expected_binary)


def test_multiple_large_regions_are_all_included():
    arr = np.full((6, 6), np.nan)
    arr[0:2, 0:2] = True  # region 1
    arr[4:6, 4:6] = True  # region 2
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_pixels=4)
    result = selector.select_regions(mask)
    # Both regions should be present
    assert np.sum(~np.isnan(result.mask)) == 8


def test_all_true_mask_returns_full_mask():
    arr = np.ones((3, 3), dtype=bool)
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_pixels=1)
    result = selector.select_regions(mask)
    assert np.all(result.mask == 1)


def test_edge_connected_region():
    arr = np.full((5, 5), np.nan)
    arr[0, :] = True  # top edge
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_pixels=5)
    result = selector.select_regions(mask)
    # The top edge should be included
    assert np.sum(~np.isnan(result.mask)) == 5


def test_min_pixels_threshold_variation():
    """Changing min_pixels should correctly include/exclude regions."""
    arr = np.full((5, 5), np.nan)
    arr[1:3, 1:3] = True  # 4 pixels
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_pixels=4)
    result = selector.select_regions(mask)
    assert np.sum(~np.isnan(result.mask)) == 4

    selector2 = ROISelector(min_pixels=5)
    result2 = selector2.select_regions(mask)
    assert np.all(np.isnan(result2.mask))


def test_touching_regions_are_separated():
    """Touching regions should be separated if not connected by structure."""
    arr = np.full((5, 5), np.nan)
    arr[1:3, 1:3] = True  # region 1
    arr[3, 1:3] = True  # region 2, touching but not connected diagonally
    mask = make_pixel_mask(arr)
    selector = ROISelector(min_pixels=2)
    result = selector.select_regions(mask)
    # Should be a single region if using 4-connectivity
    assert np.sum(~np.isnan(result.mask)) == 6

    # With 8-connectivity, all are connected
    structure = generate_binary_structure(2, 2)
    selector8 = ROISelector(min_pixels=6, structure=structure)
    result8 = selector8.select_regions(mask)
    assert np.sum(~np.isnan(result8.mask)) == 6
