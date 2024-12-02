from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np

from eitprocessing.categories import check_category
from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.datahandling.sparsedata import SparseData
from eitprocessing.features.breath_detection import BreathDetection
from eitprocessing.parameters import ParameterCalculation


@dataclass
class EELI(ParameterCalculation):
    """Compute the end-expiratory lung impedance (EELI) per breath."""

    method: Literal["breath_detection"] = "breath_detection"
    breath_detection_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        _methods = get_args(EELI.__dataclass_fields__["method"].type)
        if self.method not in _methods:
            msg = f"Method {self.method} is not valid. Use any of {', '.join(_methods)}"
            raise ValueError(msg)

    def compute_parameter(
        self,
        continuous_data: ContinuousData,
        sequence: Sequence | None = None,
        store: bool | None = None,
        result_label: str = "continuous_eelis",
    ) -> SparseData:
        """Compute the EELI for each breath in the impedance data.

        Example:
        ```
        >>> global_impedance = sequence.continuous_data["global_impedance_(raw)"]
        >>> eeli_data = EELI().compute_parameter(global_impedance)
        ```

        Args:
            continuous_data: a ContinuousData object containing impedance data.
            sequence: optional, Sequence to store the result in.
            store: whether to store the result in the sequence, defaults to `True` if a Sequence if provided.
            result_label: label of the returned SparseData object, defaults to `'continuous_eelis'`.

        Returns:
            A SparseData object with the end-expiratory values of all breaths in the impedance data.

        Raises:
            RuntimeError: If store is set to true but no sequence is provided.
            ValueError: If the provided sequence is not an instance of the Sequence dataclass.
            ValueError: If tiv_method is not one of 'inspiratory', 'expiratory', or 'mean'.
        """
        if store is None and isinstance(sequence, Sequence):
            store = True

        if store and sequence is None:
            msg = "Can't store the result if no Sequence is provided."
            raise RuntimeError(msg)

        if store and not isinstance(sequence, Sequence):
            msg = "To store the result a Sequence dataclass must be provided."
            raise ValueError(msg)

        check_category(continuous_data, "impedance", raise_=True)

        bd_kwargs = self.breath_detection_kwargs.copy()
        breath_detection = BreathDetection(**bd_kwargs)
        breaths = breath_detection.find_breaths(continuous_data)

        if not len(breaths):
            time = np.array([], dtype=float)
            values = np.array([], dtype=float)
        else:
            _, _, end_expiratory_times = zip(*breaths.values, strict=True)
            end_expiratory_indices = np.flatnonzero(np.isin(continuous_data.time, end_expiratory_times))
            time = [breath.end_time for breath in breaths.values if breath is not None]
            values = continuous_data.values[end_expiratory_indices]

        eeli_container = SparseData(
            label=result_label,
            name="End-expiratory lung impedance (EELI)",
            unit=None,
            category="impedance",
            time=time,
            description="End-expiratory lung impedance (EELI) determined on continuous data",
            parameters=self.breath_detection_kwargs,
            derived_from=[continuous_data],
            values=values,
        )
        if store:
            sequence.sparse_data.add(eeli_container)

        return eeli_container
