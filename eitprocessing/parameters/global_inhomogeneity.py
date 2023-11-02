import numpy as np
from numpy.typing import NDArray
from . import ParameterExtraction


class GlobalInhomogeneity(ParameterExtraction):
    def compute_parameter(self, pixel_dZ: NDArray) -> list:
        """Calculate global inhomogeneity

        Args:
            pixel_dZ: (n, 32, 32) np.ndarray containing dZ values per pixel for n breaths.

        Returns:
            global_inhomogeneity (float): global inhomogeneity in percentage
        """
        global_inhomogeneity = []
        for breath in range(pixel_dZ.shape[0]):
            nan_mask = np.isnan(pixel_dZ[breath, :, :])
            ventilated_area_dz = pixel_dZ[breath, ~nan_mask]
            med_dz_lung = np.median(ventilated_area_dz)
            abs_diff = [abs(pixel_dz - med_dz_lung) for pixel_dz in ventilated_area_dz]
            global_inhomogeneity.append((sum(abs_diff) / sum(ventilated_area_dz)) * 100)

        return global_inhomogeneity
