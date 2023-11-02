import numpy as np
from numpy.typing import NDArray
from . import ParameterExtraction


class RegionalVentilationDelay(ParameterExtraction):
    def __init__(
        self,
        detect_breaths_method: str = "Extreme values",
    ):
        self.detect_breaths_method = detect_breaths_method

    def compute_parameter(self, sequence, frameset_name: str) -> tuple[NDArray, list]:
        """Computes the regional ventilation delay per breath using pixel
        impedance. Also returns the regional ventilation delay inhomogeneity
        (standard deviation of RVD of all pixels)."""
        detect_breaths = DetectBreaths(method=self.detect_breaths_method)
        breaths_indices: list[tuple[int, int, int]] = detect_breaths.apply(sequence)
        start_inspiratory_indices = [indices[0] for indices in breaths_indices]
        end_inspiratory_indices = [indices[1] for indices in breaths_indices]

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
