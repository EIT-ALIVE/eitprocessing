# %%

from eitprocessing.eit_data.sentec import SentecEITData


path = "tests/test_data/20231107083148-123.zri"

eit: SentecEITData = SentecEITData.from_path(path)

# %%

from eitprocessing.plotting.animate import animate_pixel_impedance


raw = eit[280:840].variants["Slice (280-840) of <raw>"]
animate_pixel_impedance(raw.pixel_impedance_individual_offset, framerate=eit.framerate)
# %%
