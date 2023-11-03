from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from ..features import BreathDetection
from . import ParameterExtraction


@dataclass
class InhomogeneityIndex(ParameterExtraction):
    breath_detection_kwargs: dict = {}

    def compute_parameter(self, sequence, frameset_name) -> list:
        """Calculate inhomogeneity index

        Args:

        Returns:
            inhomogeneity (float): inhomogeneity index as fraction
        """
        global_impedance = sequence.framesets[frameset_name].global_impedance
        breath_detector = BreathDetection(
            sequence.framerate, **self.breath_detection_kwargs
        )
        breaths = breath_detector.find_breaths(global_impedance)

        pixel_impedance = sequence.framesets[frameset_name].pixel_values

        inhomogeneities = []
        for breath in breaths:
            insp_pixel_tiv = (
                pixel_impedance[breath.middle_index, :, :]
                - pixel_impedance[breath.start_index, :, :]
            )
            median_tiv = np.nanmedian(insp_pixel_tiv)
            abs_diff = np.abs(insp_pixel_tiv - median_tiv)
            inhomogeneities.append((np.sum(abs_diff) / np.sum(insp_pixel_tiv)))

        inhomogeneity_index = np.mean(inhomogeneities)

        return inhomogeneity_index
