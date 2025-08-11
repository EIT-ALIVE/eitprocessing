import numpy as np
from scipy.ndimage import label as nd_label

from eitprocessing.roi import PixelMask
from eitprocessing.roi.pixelmaskcollection import PixelMaskCollection


class ROISelector:
    """Class for labeling and selecting connected regions in a PixelMask."""

    def __init__(self, min_pixels: int = 10, structure: np.ndarray | None = None):
        """Initialize a ROISelector instance to identify and label regions of interest (ROIs) in a PixelMask.

        Args:
            min_pixels (int): Minimum number of pixels for a region to be considered an ROI.
                Regions smaller than this are discarded. Defaults to 10.
            structure (np.ndarray | None): Structuring element defining pixel connectivity for labeling.
                If None, uses default nearest-neighbor connectivity (e.g., 4-connectivity in 2D).
        """
        self.min_pixels = min_pixels
        self.structure = structure

    def select_regions(self, mask: PixelMask) -> PixelMaskCollection:
        """Label and select connected regions in a PixelMask and return as a new PixelMask.

        Args:
            mask (PixelMask): Input binary mask indicating pixels to be labeled.

        Returns:
            PixelMask: PixelMask object representing the union of labeled regions
                that have at least `min_pixels` pixels.
        """
        binary_array = ~np.isnan(mask.mask)
        labeled_array, num_features = nd_label(binary_array, structure=self.structure)
        masks = []
        for region_label in range(1, num_features + 1):
            region = labeled_array == region_label
            if np.sum(region) >= self.min_pixels:
                masks.append(PixelMask(region, label=f"{region_label}"))
        mask_collection = PixelMaskCollection(masks)
        return mask_collection.combine()
