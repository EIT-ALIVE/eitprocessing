import bisect
import os
from dataclasses import dataclass
from dataclasses import is_dataclass
from pprint import pprint
import numpy as np
import pytest
from typing_extensions import Self
from eitprocessing.eit_data.draeger import DraegerEITData
from eitprocessing.eit_data.eit_data_variant import EITDataVariant
from eitprocessing.eit_data.vendor import Vendor
from eitprocessing.mixins.slicing import SelectByIndex


def test_eq():
    data = DraegerEITData.from_path(
        "/home/dbodor/git/EIT-ALIVE/eitprocessing/tests/test_data/Draeger_Test3.bin"
    )
    data2 = DraegerEITData.from_path(
        "/home/dbodor/git/EIT-ALIVE/eitprocessing/tests/test_data/Draeger_Test3.bin"
    )

    data.isequivalent(data2)
