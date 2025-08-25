# %%

from matplotlib import pyplot as plt

from eitprocessing.datahandling.loading import load_eit_data

seq = load_eit_data("tests/test_data/Timpel_Test.txt", vendor="timpel")
seq.sparse_data["minvalues_(timpel)"].t[10:20]

# %%
# %%

bd = BreathDetection().find_breaths(sseq, "continuous", "global_impedance_raw")

# %%
plt.plot(sseq.continuous_data["global_impedance_raw"])
for b in bd:
    plt.axvline(b.start_index, color="g")
    plt.axvline(b.middle_index, color="r")
# %%
