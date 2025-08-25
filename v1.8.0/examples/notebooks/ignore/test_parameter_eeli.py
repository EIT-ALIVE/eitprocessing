# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: alive
#     language: python
#     name: python3
# ---

# %%
from matplotlib import pyplot as plt

from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.parameters.eeli import EELI

sequence = load_eit_data("tests/test_data/Draeger_Test.bin", vendor="draeger")
sequence = sequence.t[56600:56650]

# %%
eeli_result = EELI().compute_parameter(sequence, "global_impedance_(raw)")

# %%
gi = sequence.continuous_data["global_impedance_(raw)"]

plt.plot(gi.time, gi.values)

sd_upper = eeli_result["mean"] + eeli_result["standard deviation"]
sd_lower = eeli_result["mean"] - eeli_result["standard deviation"]

plt.axhline(eeli_result["mean"], color="red", label="Mean")
plt.axhline(eeli_result["median"], color="green", label="Median")

plt.plot(
    gi.time[eeli_result["indices"]],
    eeli_result["values"],
    "o",
    color="black",
    label="EELIs",
)

xlim = plt.xlim()
plt.fill_between(
    plt.xlim(),
    sd_upper,
    sd_lower,
    color="blue",
    alpha=0.3,
    label="Standard deviation",
)
plt.xlim(*xlim)

plt.legend()
