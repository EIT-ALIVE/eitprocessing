# %%

import numpy as np

from eitprocessing.datahandling.intervaldata import IntervalData

start_times = np.arange(10)
duration = 0.5
end_times = start_times + duration

time_ranges = zip(start_times, end_times, strict=False)
values = np.random.default_rng().random(len(start_times))

a = pd = IntervalData("label", "name", None, category="cat", intervals=time_ranges, values=values)
b = pd.select_by_time(0.4, 3.1, partial_inclusion=True)
c = pd.select_by_time(0.4, 3.1, partial_inclusion=False)
# %%
