import numpy as np
import pytest
import scipy.ndimage as ndi

from eitprocessing.roi import PixelMask
from eitprocessing.roi.roi_size_filter import FilterROIBySize


def test_basic_region_selection():
    arr = np.full((5, 5), np.nan)
    arr[1:4, 1:4] = True
    mask = PixelMask(arr, keep_zeros=True)
    selector = FilterROIBySize(min_region_size=5, connectivity=1)
    result = selector.apply(mask)
    expected_mask = np.full(arr.shape, np.nan)
    expected_mask[1:4, 1:4] = 1
    assert np.array_equal(result.mask, expected_mask, equal_nan=True)


def test_region_smaller_than_threshold_is_excluded():
    arr = np.full((5, 5), np.nan)
    arr[0, 0:2] = True
    arr[2:4, 2:5] = True
    mask = PixelMask(arr, keep_zeros=True)
    selector = FilterROIBySize(min_region_size=5, connectivity=1)
    result = selector.apply(mask)
    expected = np.zeros_like(arr, dtype=float)
    expected[2:4, 2:5] = 1
    expected[expected == 0] = np.nan
    assert np.array_equal(result.mask, expected, equal_nan=True)


def test_no_regions_above_threshold_raises():
    arr = np.full((4, 4), np.nan)
    arr[0, 0] = True
    mask = PixelMask(arr, keep_zeros=True)
    selector = FilterROIBySize(min_region_size=5, connectivity=1)
    with pytest.raises(RuntimeError, match="No regions found above min_region_size threshold"):
        selector.apply(mask)


def test_custom_connectivity():
    arr = np.full((3, 3), np.nan)
    arr[0, 0] = True
    arr[1, 1] = True
    mask = PixelMask(arr, keep_zeros=True)

    # Default (1-connectivity) — should raise
    selector_default = FilterROIBySize(min_region_size=2, connectivity=1)
    with pytest.raises(RuntimeError, match="No regions found above min_region_size threshold"):
        selector_default.apply(mask)

    # 2-connectivity — diagonal pixels connected
    selector_diag = FilterROIBySize(min_region_size=2, connectivity=2)
    result_diag = selector_diag.apply(mask)
    assert not np.all(np.isnan(result_diag.mask))

    # Custom structure
    custom_structure = ndi.generate_binary_structure(2, 2)
    selector_custom = FilterROIBySize(min_region_size=2, connectivity=custom_structure)
    result_custom = selector_custom.apply(mask)
    assert not np.all(np.isnan(result_custom.mask))


def test_empty_mask_raises():
    arr = np.full((4, 4), np.nan)
    mask = PixelMask(arr, keep_zeros=True)
    selector = FilterROIBySize(min_region_size=1, connectivity=1)
    with pytest.raises(RuntimeError, match="No regions found above min_region_size threshold"):
        selector.apply(mask)


def test_zeros_are_regions_nans_are_excluded():
    arr = np.array([[0, 1, np.nan], [0, 0.5, np.nan], [np.nan, np.nan, 0]], dtype=float)
    mask = PixelMask(arr, keep_zeros=True)
    selector = FilterROIBySize(min_region_size=3, connectivity=1)
    result_mask = selector.apply(mask).mask
    expected_mask = np.array([[1, 1, np.nan], [1, 1, np.nan], [np.nan, np.nan, np.nan]], dtype=float)
    assert np.array_equal(result_mask, expected_mask, equal_nan=True)


def test_multiple_large_regions_are_all_included():
    arr = np.full((6, 6), np.nan)
    arr[0:2, 0:2] = True
    arr[4:6, 4:6] = True
    mask = PixelMask(arr, keep_zeros=True)
    selector = FilterROIBySize(min_region_size=4, connectivity=1)
    result = selector.apply(mask)
    assert np.sum(~np.isnan(result.mask)) == 8


def test_all_true_mask_returns_full_mask():
    arr = np.ones((3, 3), dtype=bool)
    mask = PixelMask(arr, keep_zeros=True)
    selector = FilterROIBySize(min_region_size=1, connectivity=1)
    result = selector.apply(mask)
    assert np.all(result.mask == 1)


def test_edge_connected_region():
    """Test that a region around the edges of the mask is correctly identified.

    This checks that all edge pixels (top, bottom, left, right borders) are included as a single connected region.
    """
    arr = np.full((5, 5), np.nan)
    arr[0, :] = True  # top edge
    arr[-1, :] = True  # bottom edge
    arr[:, 0] = True  # left edge
    arr[:, -1] = True  # right edge
    mask = PixelMask(arr, keep_zeros=True)
    selector = FilterROIBySize(min_region_size=16, connectivity=1)
    result = selector.apply(mask)
    # There are 16 edge pixels in a 5x5 array (corners counted only once)
    assert np.sum(~np.isnan(result.mask)) == 16


def test_min_pixels_threshold_variation():
    arr = np.full((5, 5), np.nan)
    arr[1:3, 1:3] = True
    mask = PixelMask(arr, keep_zeros=True)
    selector = FilterROIBySize(min_region_size=4, connectivity=1)
    result = selector.apply(mask)
    assert np.sum(~np.isnan(result.mask)) == 4
    selector2 = FilterROIBySize(min_region_size=5, connectivity=1)
    with pytest.raises(RuntimeError, match="No regions found above min_region_size threshold"):
        selector2.apply(mask)


def test_touching_regions_are_separated():
    arr = np.full((5, 5), np.nan)
    arr[1:3, 1:3] = True
    arr[3:4, 3:5] = True
    mask = PixelMask(arr, keep_zeros=True)

    # Step 1: Check intermediate labeling for 1-connectivity
    binary_array = ~np.isnan(mask.mask)
    structure1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # 1-connectivity
    labeled_array1, num_features1 = ndi.label(binary_array, structure=structure1)
    assert num_features1 == 2, f"Expected 2 regions, got {num_features1}"
    sizes1 = [np.sum(labeled_array1 == i) for i in range(1, num_features1 + 1)]
    assert sorted(sizes1) == [2, 4], f"Unexpected sizes for 1-connectivity: {sizes1}"

    selector = FilterROIBySize(min_region_size=2, connectivity=1)
    result = selector.apply(mask)
    assert np.sum(~np.isnan(result.mask)) == 6  # total combined size

    # Step 2: Check intermediate labeling for 2-connectivity
    structure2 = np.ones((3, 3), dtype=int)  # 2-connectivity
    labeled_array2, num_features2 = ndi.label(binary_array, structure=structure2)
    assert num_features2 == 1, f"Expected 1 region for 2-connectivity, got {num_features2}"

    selector8 = FilterROIBySize(min_region_size=6, connectivity=2)
    result8 = selector8.apply(mask)
    assert np.sum(~np.isnan(result8.mask)) == 6


def test_structure_input_variants():
    """Test that selector.connectivity is the same for 1 and the equivalent custom array."""
    arr = np.full((4, 4), np.nan)
    arr[1:3, 1:3] = True
    mask = PixelMask(arr, keep_zeros=True)

    # Integer
    selector_int = FilterROIBySize(min_region_size=4, connectivity=1)
    # Array
    structure_arr = ndi.generate_binary_structure(2, 1)
    selector_arr = FilterROIBySize(min_region_size=4, connectivity=structure_arr)

    # Check that the internal structure is the same
    assert np.array_equal(selector_int.connectivity, selector_arr.connectivity)

    # Also check that the results are the same
    result_int = selector_int.apply(mask)
    result_arr = selector_arr.apply(mask)
    assert np.array_equal(result_int.mask, result_arr.mask, equal_nan=True)


def test_invalid_connectivity_none_raises():
    arr = np.full((4, 4), np.nan)
    arr[1:3, 1:3] = True
    with pytest.raises(
        ValueError, match="Unsupported connectivity type: <class 'NoneType'>. Must be an integer or numpy array."
    ):
        FilterROIBySize(min_region_size=4, connectivity=None)


def test_invalid_connectivity_value_raises():
    arr = np.full((4, 4), np.nan)
    arr[1:3, 1:3] = True
    with pytest.raises(ValueError, match="Unsupported connectivity value: 3."):
        FilterROIBySize(min_region_size=4, connectivity=3)
