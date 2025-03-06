# %%

from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.features.rate_detection import RateDetection


def test_ratedetection_works():
    path = "tests/test_data/Draeger_Test.bin"
    sequence = load_eit_data(path, vendor="draeger")
    subsequence = sequence.t[56600:56800]

    rd = RateDetection(subject_type="adult")
    rr, hr = rd.detect_respiratory_heart_rate(subsequence.eit_data["raw"])
    assert rr.values[0] == 25.0 / 60
    assert hr.values[0] == 71.0 / 60
