import numpy as np
# from ..features import BreathDetection
from . import ParameterExtraction


class EELI(ParameterExtraction):
    def __init__(
        self,
        summary_stats: dict = None,
        detect_breaths_parameters: dict = None,
    ):
        self.summary_stats = {
            "per breath": lambda x: x,
            "mean": lambda x: np.mean(x, axis=0),
            "standard deviation": lambda x: np.std(x, axis=0),
            "median": lambda x: np.median(x, axis=0),
        } | (summary_stats or {})

        self.detect_breaths_parameters = {"data_type": "continuous", "label": "global_impedance_raw"} | (detect_breaths_parameters or {})

    def compute_parameter(self, sequence, data_type: str = 'continuous', label: str = 'global_impedance_raw') -> dict | list[dict]:
        """Computes the end-expiratory lung impedance (EELI) per breath in the
        global or pixel impedance."""

        # breath_detector = BreathDetection(
        #     sequence.framerate, **self.detect_breaths_parameters
        # )
        # breaths = breath_detector.find_breaths(sequence)
        

        # _, _, end_indices = (np.array(indices) for indices in zip(breaths))

        end_indices = [6, 12, 18, 24, 32, 38]

        if data_type == "continuous":
            data = sequence.continuous_data[label]
            if data.category == "impedance":
                impedance = data.values
                eelis: np.ndarray = impedance[end_indices]

                eeli = {}
                for name, function in self.summary_stats.items():
                    eeli[name] = function(eelis)

                return eeli
            raise ValueError("The data category is not 'impedance'.")

        if data_type == "eit":
            pixel_impedance = sequence.eit_data[label].pixel_impedance

            pixel_eelis: np.ndarray = pixel_impedance[end_indices, :, :]
            eeli = {}

            for name, function in self.summary_stats.items():
                eeli[name] = function(pixel_eelis)

            return eeli

        if data_type == "sparse":
            raise ValueError("The data type cannot be sparse.")
