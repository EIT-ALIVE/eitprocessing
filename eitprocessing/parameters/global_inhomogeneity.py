import numpy as np


class GlobalInhomogeneity(ParameterExtraction):
    def __init__(
        self,
        regions: list[NDArray] | None = None,
        detect_breaths_method: str = "Extreme values",
    ):
        self.detect_breaths_method = detect_breaths_method
        self.regions = regions
