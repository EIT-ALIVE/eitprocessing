import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as nd_label

from eitprocessing.roi import PixelMask
from eitprocessing.roi.pixelmaskcollection import PixelMaskCollection


@dataclass
class FilterROIBySize:
    """Class for labeling and selecting connected regions in a PixelMask.

    This dataclass identifies and labels regions of interest (ROIs) in a PixelMask.
    You can specify the minimum region size and the connectivity structure.

    Args:
        min_region_size (int): Minimum number of pixels in a region for it to be considered an ROI.
        connectivity (str | np.ndarray): Connectivity type ("1-connectivity", "2-connectivity") or a custom structuring array.

    Connectivity:
        For 2D images, connectivity determines which pixels are considered neighbors when labeling regions.
        - "1-connectivity" (also called 4-connectivity in image processing):
                Only directly adjacent pixels (up, down, left, right) are considered neighbors.
        - "2-connectivity" (also called 8-connectivity in image processing):
                Both directly adjacent and diagonal pixels are considered neighbors.

        If a custom array is provided, it must be a boolean or integer array specifying the neighborhood structure.
        See the documentation for `scipy.ndimage.label`:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html

    Note:
        The default is "1-connectivity", which matches the default behavior of `scipy.ndimage.label`.
        `None` is not allowed for the connectivity argument; use "1-connectivity" for the default behavior.
    """

    min_region_size: int = 10
    connectivity: str | np.ndarray = "1-connectivity"

    def __post_init__(self):
        self.connectivity = self._parse_structure(self.connectivity)

    def _parse_structure(self, structure: str | np.ndarray) -> np.ndarray:
        if isinstance(structure, str):
            if structure == "1-connectivity":
                return generate_binary_structure(2, 1)
            if structure == "2-connectivity":
                return generate_binary_structure(2, 2)
            msg = f"Unknown connectivity string: {structure}"
            raise ValueError(msg)
        return structure  # assume array

    def select_regions(self, mask: PixelMask) -> PixelMask:
        """Label and select connected regions in a PixelMask and return a new PixelMask.

        Args:
            mask (PixelMask): Input binary mask indicating pixels to be labeled.

        Returns:
            PixelMask: PixelMask object representing labeled regions that have at least `min_region_size` pixels.
        """
        binary_array = ~np.isnan(mask.mask)
        labeled_array, num_features = nd_label(binary_array, connectivity=self.connectivity)
        masks = []
        for region_label in range(1, num_features + 1):
            region = labeled_array == region_label
            if np.sum(region) >= self.min_region_size:
                masks.append(PixelMask(region.astype(float), label=f"{region_label}", suppress_value_range_error=True))

        if not masks:
            warnings.warn("No regions found above min_region_size threshold.", UserWarning)
            empty_mask = np.full(mask.mask.shape, np.nan)
            return PixelMask(empty_mask)

        mask_collection = PixelMaskCollection(masks)
        return mask_collection.combine()
