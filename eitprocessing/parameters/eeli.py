import numpy as np
from numpy.typing import NDArray
from . import ParameterExtraction
from ._temp_class import DetectBreaths


class EELI(ParameterExtraction):
    def __init__(
        self,
        regions: list[NDArray] | None = None,
        detect_breaths_method: str = "Extreme values",
    ):
        self.detect_breaths_method = detect_breaths_method
        self.regions = regions

    def compute_parameter(
        self, sequence, frameset_name: str
    ) -> tuple[list, NDArray] | list[tuple[list, NDArray]]:
        """Computes the end-expiratory lung impedance (EELI) per breath in the
        global impedance."""
        detect_breaths = DetectBreaths(method=self.detect_breaths_method)
        breaths_indices: list[tuple[int, int, int]] = detect_breaths.apply(sequence)
        end_expiratory_indices = [indices[2] for indices in breaths_indices]

        if self.regions is None:
            global_impedance = sequence.framesets[frameset_name].global_impedance
            global_eeli: NDArray = global_impedance[end_expiratory_indices]
            return (end_expiratory_indices, global_eeli)
        else:
            regional_eeli = []
            pixel_impedance = sequence.framesets[frameset_name].pixel_values
            for region in self.regions:
                regional_pixel_impedance = np.matmul(pixel_impedance, region)
                regional_impedance = np.nansum(regional_pixel_impedance, axis=(0, 1))
                regional_eeli.append(
                    (end_expiratory_indices, regional_impedance[end_expiratory_indices])
                )
            return regional_eeli