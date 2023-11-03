import numpy as np
from numpy.typing import NDArray
from ..features import BreathDetection
from . import ParameterExtraction


@dataclass
class RegionalVentilationDelay(ParameterExtraction):
    breath_detection_kwargs: dict = {}

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

    # interpolated_signals = []
    # for start, end in zip(start_insp, end_insp):

    #     num_timepoints = end - start

    #     # By dividing by (num_timepoints - 1) the time axis starts at 0 and ends at 1
    #     norm_time = np.arange(num_timepoints) / (num_timepoints - 1)

    #     inspiration = all_pixels[start:end, :, :]

    #     def interpolate(y):
    #         return interp1d(norm_time, y, kind=INTERPOLATION_KIND)(COMMON_TIME)

    #     # This applies interpolation for every pixel
    #     interpolated_imp = np.apply_along_axis(interpolate, 0, inspiration)
    #     interpolated_signals.append(interpolated_imp)

    # # Mean inspiration for each pixel
    # pix_matrix = np.mean(interpolated_signals, axis=0)

    ## Code below should be adjusted to ALIVE format
    # offset = pix_matrix[0,:,:]
    # pix_matrix -= offset
    # total_imp = pix_matrix.max(axis=0)
    # norm_pix_imp = (pix_matrix - 0)/ (total_imp-0)
    # rvd = np.argmax(norm_pix_imp > 0.4, axis=0)
    # non_zero_rvd = rvd[rvd != 0]
    # vmin = np.min(non_zero_rvd)
    # vmax = np.max(rvd)

    # plt.title(f"Regional ventilation delay map at {timepoint}")
    # plt.imshow(rvd, vmin=vmin, vmax=vmax)
    # plt.savefig(results_path / f'Regional ventilation delay map - patient {patient_ID} - {timepoint}.png', dpi=300)
    # plt.close()

    # rvdi = np.std(non_zero_rvd)

    # for row in range(0,32):
    #     for col in range (0,32):
    #         if isinstance(pix_matrix[row,col], np.ndarray):
    #             plt.plot(COMMON_TIME, norm_pix_imp[:,row,col])
    #             plt.title(f'Normalized pixel impedance of all ventilated pixels at {timepoint}')
    #             plt.xlabel('Normalized time')
    #             plt.ylabel('Normalized impedance')
    # plt.savefig(results_path / f'Normalized pixel impedance of all ventilated pixels - patient {patient_ID} - {timepoint}.png', dpi=300)
    # plt.close()

    # return rvdi
