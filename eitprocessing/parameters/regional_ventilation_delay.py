import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from ..features import BreathDetection
from . import ParameterExtraction


@dataclass
class RegionalVentilationDelay(ParameterExtraction):
    breath_detection_kwargs: dict = {}
    interpolation_kind = "cubic"  # Type of interpolation used for normalizing all inspirations between 0 and 1
    common_time = np.linspace(0, 1, 101)  # Common time axis to which is normalized
    summary_stats: dict[str, Callable[[NDArray], float]] = {
        "mean": np.mean,
        "standard deviation": np.std,
        "median": np.median,
    }

    def compute_parameter(self, sequence, frameset_name: str) -> tuple[NDArray, dict]:
        """Computes the mean regional ventilation delay over all breaths using pixel
        impedance. Also returns the regional ventilation delay inhomogeneity
        (standard deviation of RVD of all pixels) in a dictionary according to
        the required summary statistics."""
        global_impedance = sequence.framesets[frameset_name].global_impedance
        breath_detector = BreathDetection(
            sequence.framerate, **self.breath_detection_kwargs
        )
        breaths = breath_detector.find_breaths(global_impedance)

        pixel_impedance = sequence.framesets[frameset_name].pixel_values

        regional_ventilation_delays = []
        rvd_inhomogeneities = []
        for breath in breaths:
            num_timepoints = breath.middle_index - breath.start_index

            # By dividing by (num_timepoints - 1) the time axis starts at 0 and ends at 1
            norm_time = np.arange(num_timepoints) / (num_timepoints - 1)

            pixel_inspiration = pixel_impedance[
                breath.start_index : breath.middle_index, :, :
            ]

            def interpolate(y):
                return interp1d(norm_time, y, kind=self.interpolation_kind)(
                    self.common_time
                )

            # This applies interpolation for every pixel
            interpolated_inspiration = np.apply_along_axis(
                interpolate, 0, pixel_inspiration
            )
            interpolated_inspirations.append(interpolated_inspiration)

            offset = interpolated_inspirations[0, :, :]
            interpolated_inspirations -= offset
            max_impedance = interpolated_inspirations.max(axis=0)
            normalized_pixel_tiv = (interpolated_inspirations - 0) / (max_impedance - 0)
            rvd = np.nanargmax(normalized_pixel_tiv > 0.4, axis=0)
            regional_ventilation_delays.append(rvd)
            rvd_inhomogeneities.append(np.std(rvd))

        mean_rvd = np.mean(regional_ventilation_delays, axis=0)

        rvd_inhomogeneity = {}
        for name, function in self.summary_stats.items():
            rvd_inhomogeneity[name] = function(rvd_inhomogeneities)

        return (mean_rvd, rvd_inhomogeneity)
