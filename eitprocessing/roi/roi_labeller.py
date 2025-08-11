import numpy as np
from scipy.ndimage import label as nd_label

from eitprocessing.roi import PixelMask
from eitprocessing.roi.pixelmaskcollection import PixelMaskCollection


class ROILabeller:
    """Class for labeling connected regions in a PixelMask."""

    def __init__(self, min_pixels: int = 10, structure: np.ndarray | None = None):
        """Initialize a ROILabeller instance to identify and label regions of interest (ROIs) in a PixelMask.

        Args:
            min_pixels (int): Minimum number of pixels for a region to be considered an ROI.
                Regions smaller than this are discarded. Defaults to 10.
            structure (np.ndarray | None): Structuring element defining pixel connectivity for labeling.
                If None, uses default nearest-neighbor connectivity (e.g., 4-connectivity in 2D).
        """
        self.min_pixels = min_pixels
        self.structure = structure

    def label_regions(self, mask: PixelMask) -> PixelMaskCollection:
        """Label connected regions in a PixelMask and return as separate PixelMask objects in a PixelMaskCollection.

        Args:
            mask (PixelMask): Input binary mask indicating pixels to be labeled.

        Returns:
            PixelMaskCollection: Collection of PixelMask objects representing labeled regions
                that have at least `min_pixels` pixels. Each PixelMask has a unique integer label.
        """
        binary_array = ~np.isnan(mask.mask)
        labeled_array, num_features = nd_label(binary_array, structure=self.structure)
        masks = []
        for region_label in range(1, num_features + 1):
            region = labeled_array == region_label
            if np.sum(region) >= self.min_pixels:
                masks.append(PixelMask(region, label=region_label))
        return PixelMaskCollection(masks)
