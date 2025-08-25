# %%
from matplotlib import pyplot as plt

from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.features.detect_respiratory_heart_rate import DetectRespiratoryHeartRates

# %%
seq = load_eit_data(
    "/Users/peter/Library/Mobile Documents/com~apple~CloudDocs/Downloads/20220217115837.zri", vendor="sentec"
)
drhr = DetectRespiratoryHeartRates(seq.eit_data["raw"].sample_frequency, subject_type="adult")

# %%
sseq = seq.t[400:600]
rr, hr = drhr.apply(sseq.eit_data["raw"].pixel_impedance)

# %%
print(f"{rr*60=}")
print(f"{hr*60=}")
# %%
plt.plot(sseq.time, sseq.eit_data["raw"].calculate_global_impedance())

# %%
