import warnings

import numpy as np
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as nd_label

from eitprocessing.roi import PixelMask
from eitprocessing.roi.pixelmaskcollection import PixelMaskCollection


class ROISelector:
    """Class for labeling and selecting connected regions in a PixelMask."""

    def __init__(self, min_pixels: int = 10, structure: str | np.ndarray | None = None):
        """Initialize a ROILabeller instance to identify and label regions of interest (ROIs) in a PixelMask.

        Args:
        min_region_size (int): Minimum number of pixels in a region for it to be considered an ROI.
        structure (str | np.ndarray | None): Connectivity type ("4-connectivity", "8-connectivity") or custom array.
        """
        self.min_pixels = min_pixels
        self.structure = self._parse_structure(structure)

    def _parse_structure(self, structure: str | np.ndarray | None) -> np.ndarray | None:
        if structure is None:
            return None  # default nearest-neighbor
        if isinstance(structure, str):
            if structure == "4-connectivity":
                return generate_binary_structure(2, 1)
            if structure == "8-connectivity":
                return generate_binary_structure(2, 2)
            msg = f"Unknown connectivity string: {structure}"
            raise ValueError(msg)
        return structure  # assume array

    def select_regions(self, mask: PixelMask) -> PixelMask:
        """Label and select connected regions in a PixelMask and return a new PixelMask.

        Args:
            mask (PixelMask): Input binary mask indicating pixels to be labeled.

        Returns:
            PixelMask: PixelMask object representing labeled regions that have at least `min_pixels` pixels.
        """
        binary_array = ~np.isnan(mask.mask)
        labeled_array, num_features = nd_label(binary_array, structure=self.structure)
        masks = []
        for region_label in range(1, num_features + 1):
            region = labeled_array == region_label
            if np.sum(region) >= self.min_pixels:
                masks.append(PixelMask(region.astype(float), label=f"{region_label}", suppress_value_range_error=True))

        if not masks:
            warnings.warn("No regions found above min_pixels threshold.", UserWarning)
            empty_mask = np.full(mask.mask.shape, np.nan)
            return PixelMask(empty_mask)

        mask_collection = PixelMaskCollection(masks)
        return mask_collection.combine()
