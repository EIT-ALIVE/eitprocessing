# %%
from matplotlib.cbook import file_requires_unicode

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.rate_detection import RateDetection
from tests.conftest import draeger1, draeger2, timpel1

MINUTE = 60

# %%


sequence: Sequence = timpel1.__pytest_wrapped__.obj()  # load sequence using pytest fixture
eit_data: EITData = sequence.eit_data["raw"]

sub_data = eit_data  # .t[56650:56760]

from matplotlib import pyplot as plt

plt.plot(sub_data.time, sub_data.pixel_impedance[:, 16, 16], label="Pixel Impedance")

rd = RateDetection("adult")
rr, hr = rd.apply(sub_data, captures=(captures := {}))
fig = rd.plotting.plot(**captures)
fig.show()

# %%

from matplotlib import pyplot as plt

from tests.test_rate_detection import signal_factory

signal = signal_factory.__pytest_wrapped__.obj()(
    high_power_frequencies=(true_rr := 0.20,),
    low_power_frequencies=(true_hr := 0.9090909090909090,),
    duration=120,
    sample_frequency=20.0,
    low_power_amplitude=0.01,
    noise_amplitude=0.1,
    captures=(captures_ := {}),
    high_frequency_scale_factor=1.4,
    low_frequency_scale_factor=0.7,
)
rr, hr = RateDetection("adult", refine_estimated_frequency=False, welch_window=90.0, welch_overlap=0).apply(
    signal, captures=(captures := {})
)
rr_, hr_ = RateDetection("adult", refine_estimated_frequency=True, welch_window=90.0).apply(
    signal, captures=(captures := {})
)
print(rr, rr_, true_rr)
print(round(hr * MINUTE, 2), round(hr_ * MINUTE, 2), round(true_hr * MINUTE, 2))
fig = rd.plotting.plot(**captures)
fig.show()

plt.figure()
plt.plot(captures_["time"], captures_["high_power_signal"], label="High Power Signal")

# %%

high_power_frequency = 0.325
low_power_frequency = 1.33333333333333
frequency_multipliers = (1, 1.2, 1.4)
high_power_frequencies = tuple(high_power_frequency * m for m in frequency_multipliers)
low_power_frequencies = tuple(low_power_frequency * m for m in frequency_multipliers)


signal = signal_factory.__pytest_wrapped__.obj()(
    high_power_frequencies=high_power_frequencies,
    low_power_frequencies=low_power_frequencies,
    duration=60.0,
    sample_frequency=20.0,
    shape=(32, 32),
    low_power_amplitude=0.01,
    noise_amplitude=0.1,
)

rd = RateDetection("adult", refine_estimated_frequency=True)
rr, hr = rd.apply(signal, suppress_edge_case_warning=True, suppress_length_warnings=True)


print(f"""
Respiratory rate:
    base: {", ".join(f"{f * MINUTE:.1f}" for f in high_power_frequencies)} breaths/min
    estimated: {rr * MINUTE:.1f} breaths/min

Heart rate:
    base: {", ".join(f"{f * MINUTE:.1f}" for f in low_power_frequencies)} breaths/min
    estimated: {hr * MINUTE:.1f} beats/min
""")

assert np.any(np.isclose(rr, high_power_frequencies, rtol=0.05, atol=0.5 / MINUTE))
assert np.any(np.isclose(hr, low_power_frequencies, rtol=0.05, atol=0.5 / MINUTE))
