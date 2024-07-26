import warnings

import numpy as np
import pytest

from eitprocessing.datahandling.continuousdata import ContinuousData


def test_sample_frequency_deprecation_warning():
    n = 100
    sample_frequency = 10
    time = np.arange(n) / sample_frequency
    values = np.arange(n)

    with pytest.warns(DeprecationWarning):
        ContinuousData(
            "label",
            "name",
            "unit",
            "category",
            time=time,
            values=values,
        )

    with pytest.warns(DeprecationWarning):
        ContinuousData(
            "label",
            "name",
            "unit",
            "category",
            time=time,
            values=values,
            sample_frequency=None,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # upgrades DeprecationWarning to raised exception
        ContinuousData(
            "label",
            "name",
            "unit",
            "category",
            time=time,
            values=values,
            sample_frequency=sample_frequency,
        )
