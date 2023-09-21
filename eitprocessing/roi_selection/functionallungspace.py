import numpy as np
from . import ROISelection


class FunctionalLungSpace(ROISelection):
    def __init__(
        self,
        threshold: float,
        min_output_size: int = 0,
        min_cluster_size: int = 0,
    ):
        self.threshold = threshold
        self.min_output_size = min_output_size
        self.min_cluster_size = min_cluster_size

    def find_ROI(
        self,
        data: np.ndarray,
    ):
        max_pixel = np.nanmax(data, axis=-1)
        min_pixel = np.nanmin(data, axis=-1)
        pixel_amplitude = max_pixel - min_pixel
        max_pixel_amplitude = np.max(pixel_amplitude)

        output = pixel_amplitude > (max_pixel_amplitude * self.threshold)
        output = self.minimal_cluster(output)

        return output
