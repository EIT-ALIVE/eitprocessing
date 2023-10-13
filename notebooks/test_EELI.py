# %%

from eitprocessing.binreader import Sequence
from eitprocessing.parameters._temp_class import DetectBreaths
from eitprocessing.parameters.eeli import EELI


sequence = Sequence.from_path("tests/test_data/Draeger_Test_4.bin", vendor="draeger")

# %%

db = DetectBreaths()
breaths = db.apply(sequence)
# %%

eeli_detector = EELI()
ind, eeli = eeli_detector.compute_parameter(sequence, "raw")

# %%
from matplotlib import pyplot as plt

plt.plot(sequence.framesets["raw"].global_impedance)
plt.plot(ind, eeli, color="r")
# %%
