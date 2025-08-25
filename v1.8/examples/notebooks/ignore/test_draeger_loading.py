# %%

from eitprocessing.datahandling.loading import load_eit_data

seq = load_eit_data("tests/test_data/Draeger_Test.bin", vendor="draeger")
