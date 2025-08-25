# %%

from eitprocessing.datahandling.loading import load_eit_data

seq = load_eit_data("tests/test_data/Timpel_Test.txt", vendor="timpel")

# %%

display(seq.interval_data["breaths_(timpel)"].time_ranges)
display(seq.t[3:10].interval_data["breaths_(timpel)"].time_ranges)

# %%
