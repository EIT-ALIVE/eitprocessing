# %%
import numpy as np

from eitprocessing.datahandling.loading import load_eit_data

cutoff = 1000

timpel_file = "tests/test_data/Timpel_Test.txt"

timpel = load_eit_data(timpel_file, vendor="timpel")
timpel_part_1 = load_eit_data(timpel_file, vendor="timpel", max_frames=cutoff)
timpel_part_2 = load_eit_data(timpel_file, vendor="timpel", first_frame=cutoff)

# %%
assert len(timpel) == len(timpel_part_1) + len(timpel_part_2)
assert len(timpel_part_1) == cutoff

timpel_sliced = timpel[:cutoff]
timpel_sliced2 = timpel[cutoff:]


# %%

for key in vars(timpel_sliced2):
    print(key)
    print(getattr(timpel_sliced2, key))
    print(getattr(timpel_part_1, key))

# %%

assert timpel_sliced == timpel_part_1

for key in vars(timpel_sliced2.eit_data["raw"]):
    print(key)
    print(getattr(timpel_sliced2.eit_data["raw"], key))
    print(getattr(timpel_part_2.eit_data["raw"], key))

assert timpel_sliced2 == timpel_part_2

# %%

cutoff = 1000
draeger_file2 = "tests/test_data/Draeger_Test.bin"
draeger2 = load_eit_data(draeger_file2, vendor="draeger")
draeger2_part1 = load_eit_data(draeger_file2, "draeger", max_frames=cutoff, label="draeger_part_1")
draeger2_part2 = load_eit_data(draeger_file2, "draeger", first_frame=cutoff, label="draeger_part_2")

sliced_draeger2 = draeger2[cutoff:]

for key in vars(sliced_draeger2.eit_data["raw"]):
    print(key)
    print(getattr(sliced_draeger2.eit_data["raw"], key))
    print(getattr(draeger2_part2.eit_data["raw"], key))

assert draeger2_part1 == draeger2[:cutoff]
assert draeger2_part2 == draeger2[cutoff:]


# %%
