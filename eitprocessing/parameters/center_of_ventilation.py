import numpy as np
from numpy.typing import NDArray
from ..features import BreathDetection
from . import ParameterExtraction


@dataclass
class CenterOfVentilation(ParameterExtraction):
    breath_detection_kwargs: dict = {}

    def compute_parameter(self, sequence, frameset_name: str) -> tuple[NDArray, float]:
        global_impedance = sequence.framesets[frameset_name].global_impedance
        breath_detector = BreathDetection(
            sequence.framerate, **self.breath_detection_kwargs
        )
        breaths = breath_detector.find_breaths(global_impedance)

        pixel_impedance = sequence.framesets[frameset_name].pixel_values

        centers_of_ventilation_y = []
        center_of_ventilation_y_positions = []
        centers_of_ventilation_x = []
        center_of_ventilation_x_positions = []

        for breath in breaths:
            insp_pixel_tiv = (
                pixel_impedance[breath.middle_index, :, :]
                - pixel_impedance[breath.start_index, :, :]
            )
            n_rows = insp_pixel_tiv.shape[0]
            n_cols = insp_pixel_tiv.shape[1]

            weights_y = (np.arange(n_rows) + 0.5) / n_rows
            weights_x = (np.arange(n_cols) + 0.5) / n_cols

            y_direction_weighted = insp_pixel_tiv * weights_y[:, np.newaxis]
            y_direction_weighted_sum = np.sum(y_direction_weighted, axis=None)

            x_direction_weighted = insp_pixel_tiv * weights_x[:, np.newaxis]
            x_direction_weighted_sum = np.sum(x_direction_weighted, axis=None)

            insp_pixel_tiv_sum = np.sum(insp_pixel_tiv, axis=None)

            cov_y = y_direction_weighted_sum / insp_pixel_tiv_sum
            centers_of_ventilation_y.append(cov_y)
            center_of_ventilation_y_positions.append(
                cov_y * n_rows
            )  # pay attention to how pixel rows are displayed (centered on tick or tick at start of row)

            cov_x = x_direction_weighted_sum / insp_pixel_tiv_sum
            centers_of_ventilation_x.append(cov_x)
            center_of_ventilation_x_positions.append(cov_x * n_cols)

        center_of_ventilation_y = np.mean(centers_of_ventilation_y)
        center_of_ventilation_y_position = np.mean(center_of_ventilation_y_positions)
        center_of_ventilation_x = np.mean(centers_of_ventilation_x)
        center_of_ventilation_x_position = np.mean(center_of_ventilation_x_positions)

        return (
            (center_of_ventilation_x, center_of_ventilation_y),
            (center_of_ventilation_x_position, center_of_ventilation_y_position),
        )
