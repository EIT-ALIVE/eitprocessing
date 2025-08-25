import warnings

import numpy as np
import pytest

from eitprocessing.datahandling.continuousdata import ContinuousData


def test_sample_frequency_deprecation_warning():
    n = 100
    sample_frequency = 10
    time = np.arange(n) / sample_frequency
    values = np.arange(n)

    with pytest.warns(DeprecationWarning, match="`sample_frequency` is set to `None`"):
        ContinuousData(
            "label",
            "name",
            "unit",
            "category",
            time=time,
            values=values,
        )

    with pytest.warns(DeprecationWarning, match="`sample_frequency` is set to `None`"):
        ContinuousData(
            "label",
            "name",
            "unit",
            "category",
            time=time,
            values=values,
            sample_frequency=None,
        )

    with warnings.catch_warnings(record=True) as w:
        ContinuousData(
            "label",
            "name",
            "unit",
            "category",
            time=time,
            values=values,
            sample_frequency=sample_frequency,
        )
        assert len(w) == 0, "No warnings should be raised when sample_frequency is provided"
