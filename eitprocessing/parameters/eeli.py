from collections.abc import Callable

import numpy as np

from eitprocessing.features import BreathDetection
from eitprocessing.parameters import ParameterExtraction


class EELI(ParameterExtraction):
    def __init__(
        self,
        regions: list[np.ndarray] | None = None,
        detect_breaths_method: str = "Extreme values",
        summary_stats: dict[str, Callable[[np.ndarray], float]] | None = None,
    ):
        if summary_stats is None:
            summary_stats = {
                "per breath": lambda x: x,
                "mean": np.mean,
                "standard deviation": np.std,
                "median": np.median,
            }
        self.detect_breaths_method = detect_breaths_method
        self.regions = regions
        self.summary_stats = summary_stats

    def compute_parameter(self, sequence, frameset_name: str) -> dict | list[dict]:
        """Computes the end-expiratory lung impedance (EELI) per breath in the
        global impedance.
        """
        breath_detector = BreathDetection(
            sequence.framerate,
            **self.breath_detection_kwargs,
        )
        breaths = breath_detector.find_breaths(global_impedance)

        _, _, end_indices = (np.array(indices) for indices in zip(breaths, strict=False))

        if self.regions is None:
            global_impedance = sequence.framesets[frameset_name].global_impedance
            global_eelis: np.ndarray = global_impedance[end_indices]

            global_eeli = {}
            for name, function in self.summary_stats.items():
                global_eeli[name] = function(global_eelis)

            return global_eeli
        else:
            pixel_impedance = sequence.framesets[frameset_name].pixel_values
            regional_eeli = []
            for region in self.regions:
                regional_pixel_impedance = np.matmul(pixel_impedance, region)
                regional_impedance = np.nansum(regional_pixel_impedance, axis=(0, 1))
                regional_eelis: np.ndarray = regional_impedance[end_indices]
                results = {}
                for name, function in self.summary_stats.items():
                    results[name] = function(regional_eelis)
                regional_eeli.append(results)

            return regional_eeli
