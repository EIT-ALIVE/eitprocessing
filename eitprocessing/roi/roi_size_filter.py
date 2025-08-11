from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as nd_label

from eitprocessing.roi import PixelMask
from eitprocessing.roi.pixelmaskcollection import PixelMaskCollection


@dataclass(frozen=True, kw_only=True)
class FilterROIBySize:
    """Class for labeling and selecting connected regions in a PixelMask.

    This dataclass identifies and labels regions of interest (ROIs) in a PixelMask.
    You can specify the minimum region size and the connectivity structure.

    Connectivity:
        For 2D images, connectivity determines which pixels are considered neighbors when labeling regions.
        - "1-connectivity" (also called 4-connectivity in image processing):
                Only directly adjacent pixels (up, down, left, right) are considered neighbors.
        - "2-connectivity" (also called 8-connectivity in image processing):
                Both directly adjacent and diagonal pixels are considered neighbors.

        The default value is "1-connectivity". 

        If a custom array is provided, it must be a boolean or integer array specifying the neighborhood structure.
        See the documentation for `scipy.ndimage.label`:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html


    Args:
        min_region_size (int): Minimum number of pixels in a region for it to be considered an ROI.
        connectivity (Literal["1-connectivity", "2-connectivity"] | np.ndarray):
            Connectivity type ("4-connectivity", "8-connectivity") or custom array.
    """

    min_region_size: int = 10
    connectivity: Literal["1-connectivity", "2-connectivity"] | np.ndarray = "1-connectivity"

    def __post_init__(self):
        object.__setattr__(
            self, "connectivity", self._parse_connectivity(self.connectivity)
        )  # required with frozen objects

    def _parse_connectivity(self, connectivity: str | np.ndarray) -> np.ndarray:
        if isinstance(connectivity, str):
            if connectivity == "1-connectivity":
                return generate_binary_structure(2, 1)
            if connectivity == "2-connectivity":
                return generate_binary_structure(2, 2)
            msg = (
                f"Unsupported connectivity string: {connectivity}. "
                "Change to '1-connectivity' or '2-connectivity' or input a custom structure"
            )
            raise ValueError(msg)
        if isinstance(connectivity, np.ndarray):
            return connectivity
        msg = f"Unsupported connectivity type: {type(connectivity)}. Must be a string or numpy array."
        raise ValueError(msg)

    def apply(self, mask: PixelMask) -> PixelMask:
        """Identify connected regions in a PixelMask, filter them by size, and return a combined mask.

        This method:
        1. Converts the input PixelMask into a binary representation where all non-NaN values
            are treated as part of a region and NaNs are excluded.
        2. Labels connected components using the specified connectivity structure.
        3. Keeps only those connected regions whose pixel count is greater than or equal
            to `self.min_region_size`.
        4. Combines the remaining regions into a single PixelMask.

        Args:
            mask (PixelMask):
                Input mask where non-NaN pixels are considered valid region pixels.
                NaNs are treated as excluded/background.

        Returns:
            PixelMask:
                A new PixelMask representing the union of all regions that meet the
                `min_region_size` criterion.

        Raises:
            RuntimeError:
                If no connected regions meet the size threshold (e.g., mask is empty,
                all regions are too small, or connectivity is too restrictive).
        """
        binary_array = ~np.isnan(mask.mask)
        labeled_array, num_features = nd_label(binary_array, structure=self.connectivity)
        masks = []
        for region_label in range(1, num_features + 1):
            region = labeled_array == region_label
            if np.sum(region) >= self.min_region_size:
                masks.append(PixelMask(region, suppress_value_range_error=True))

        if not masks:
            msg = (
                "No regions found above min_region_size threshold. "
                "This can occur if your input mask is empty, all regions are too small,"
                " or your connectivity is too restrictive."
            )
            raise RuntimeError(msg)

        mask_collection = PixelMaskCollection(masks)
        return mask_collection.combine()
