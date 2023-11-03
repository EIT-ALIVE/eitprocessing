import numpy as np
from numpy.typing import NDArray
from ..features import BreathDetection
from . import ParameterExtraction


@dataclass
class RespiratoryDurations(ParameterExtraction):
    breath_detection_kwargs: dict = {}
    summary_stats: dict[str, Callable[[NDArray], float]] = {
        "mean": np.mean,
        "standard deviation": np.std,
        "median": np.median,
    }

    def compute_parameter(self, sequence, frameset_name: str) -> tuple[NDArray, float]:
        global_impedance = sequence.framesets[frameset_name].global_impedance
        breath_detector = BreathDetection(
            sequence.framerate, **self.breath_detection_kwargs
        )
        breaths = breath_detector.find_breaths(global_impedance)

        start_indices, middle_indices, end_indices = (
            np.array(indices) for indices in zip(breaths)
        )

        inspiratory_durations = (middle_indices - start_indices) / sequence.framerate
        expiratory_durations = (end_indices - middle_indices) / sequence.framerate
        breath_durations = inspiratory_durations + expiratory_durations
        respiratory_rates = 60 / breath_durations

        respiratory_durations = {
            "inspiratory_duration": {},
            "expiratory_duration": {},
            "breath_duration": {},
            "respiratory_rate": {},
        }
        for name, function in self.summary_stats.items():
            respiratory_durations["inspiratory_duration"][name] = function(
                inspiratory_durations
            )
            respiratory_durations["expiratory_duration"][name] = function(
                expiratory_durations
            )
            respiratory_durations["breath_duration"][name] = function(breath_durations)
            respiratory_durations["respiratory_rate"][name] = function(
                respiratory_rates
            )

        return respiratory_durations
