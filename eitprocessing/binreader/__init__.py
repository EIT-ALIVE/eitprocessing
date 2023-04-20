"""Loading and processing binary EIT data from the Dräger Pulmovista 500"""

__version__ = "0.1"

from .reader import Reader
from .frameset import Frameset
from .sequence import Sequence
from .phases import MaxValue, MinValue
