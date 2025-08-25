# %%

from pathlib import Path

from eitprocessing.datahandling.loading import load_eit_data

paths = Path("/Volumes/Beademing/COVID-19/Studies/Stable period selection/Data/").glob("C032-b*.bin")
paths = sorted(paths)


# %%

load_eit_data(paths[0:2], vendor="draeger")

# %%
