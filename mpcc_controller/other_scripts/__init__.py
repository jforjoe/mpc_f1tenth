"""
MPCC Controller for F1TENTH Autonomous Racing

Model Predictive Contouring Control implementation based on ETH Zurich's work.
Reference: arXiv:1711.07300v1
"""

__version__ = '1.0.0'
__author__ = 'Your Name'
__license__ = 'MIT'

from .track_map import TrackMap
from .vehicle_model import VehicleModel
from .mpcc_controller import MPCCController

__all__ = ['TrackMap', 'VehicleModel', 'MPCCController']
