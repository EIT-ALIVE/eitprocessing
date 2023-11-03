import numpy as np
from numpy.typing import NDArray
from ..features import BreathDetection
from . import ParameterExtraction


@dataclass
class RegionalVentilationDelay(ParameterExtraction):
    breath_detection_kwargs: dict = {}
    interpolation_kind = "cubic"  # Type of interpolation used for normalizing all inspirations between 0 and 1
    common_time = np.linspace(0, 1, 101)  # Common time axis to which is normalized

    def compute_parameter(self, sequence, frameset_name: str) -> tuple[NDArray, list]:
        """Computes the regional ventilation delay per breath using pixel
        impedance. Also returns the regional ventilation delay inhomogeneity
        (standard deviation of RVD of all pixels)."""
        global_impedance = sequence.framesets[frameset_name].global_impedance
        breath_detector = BreathDetection(
            sequence.framerate, **self.breath_detection_kwargs
        )
        breaths = breath_detector.find_breaths(global_impedance)

        pixel_impedance = sequence.framesets[frameset_name].pixel_values

        interpolated_inspirations = []
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

        # Mean inspiration for each pixel
        mean_pixel_inspiration = np.mean(interpolated_inspirations, axis=0)

        offset = mean_pixel_inspiration[0, :, :]
        mean_pixel_inspiration -= offset
        max_impedance = mean_pixel_inspiration.max(axis=0)
        normalized_pixel_tiv = (mean_pixel_inspiration - 0) / (max_impedance - 0)
        rvd = np.argmax(normalized_pixel_tiv > 0.4, axis=0)
        non_zero_rvd = rvd[rvd != 0]
        vmin = np.min(non_zero_rvd)
        vmax = np.max(rvd)

    # plt.title(f"Regional ventilation delay map at {timepoint}")
    # plt.imshow(rvd, vmin=vmin, vmax=vmax)
    # plt.savefig(results_path / f'Regional ventilation delay map - patient {patient_ID} - {timepoint}.png', dpi=300)
    # plt.close()

    # rvdi = np.std(non_zero_rvd)

    # for row in range(0,32):
    #     for col in range (0,32):
    #         if isinstance(mean_pixel_inspiration[row,col], np.ndarray):
    #             plt.plot(COMMON_TIME, normalized_pixel_tiv[:,row,col])
    #             plt.title(f'Normalized pixel impedance of all ventilated pixels at {timepoint}')
    #             plt.xlabel('Normalized time')
    #             plt.ylabel('Normalized impedance')
    # plt.savefig(results_path / f'Normalized pixel impedance of all ventilated pixels - patient {patient_ID} - {timepoint}.png', dpi=300)
    # plt.close()

    # return rvdi
