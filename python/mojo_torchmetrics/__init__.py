"""Classification and regression metrics accelerated by Mojo."""

from .metric import Metric
from . import classification as classification
from . import regression as regression
from .classification import *
from .regression import *
from . import functional

__version__ = "0.1.0"

__all__ = ["Metric", "functional"]
__all__ += classification.__all__
__all__ += regression.__all__
