# %%

import copy

from eitprocessing.datahandling.loading import load_eit_data

seq = load_eit_data("tests/test_data/Draeger_Test_4.bin", vendor="draeger")
bc = copy.deepcopy(seq.eit_data["raw"])
bc.label = "bc"
bc.pixel_impedance = bc.pixel_impedance - bc.pixel_baseline
seq.eit_data.add(bc)

# %%
