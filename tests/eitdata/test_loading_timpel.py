import pytest

from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence


@pytest.mark.parametrize("sequence", ["timpel_healthy_volunteer_1", "timpel_healthy_volunteer_2"], indirect=True)
def test_loading_timpel(
    sequence: Sequence,
):
    assert isinstance(sequence, Sequence)
    assert isinstance(sequence.eit_data["raw"], EITData)
    assert sequence.eit_data["raw"].vendor == Vendor.TIMPEL

    for key in ["global_impedance_(raw)", "airway_pressure_(timpel)", "flow_(timpel)", "volume_(timpel)"]:
        assert key in sequence.continuous_data, f"Missing continuous data key: {key}"

    for key in ["breaths_(timpel)"]:
        assert key in sequence.interval_data, f"Missing interval data key: {key}"

    for key in ["minvalues_(timpel)", "maxvalues_(timpel)", "qrscomplexes_(timpel)"]:
        assert key in sequence.sparse_data, f"Missing sparse data key: {key}"

    loaded_using_enum_vendor = load_eit_data(sequence.eit_data["raw"].path, vendor=Vendor.TIMPEL, label="timpel")
    assert sequence == loaded_using_enum_vendor
