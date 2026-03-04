"""
This is a package that can be used for energy and gradient calculation
by external programs on the cluster. The programs include:
- Gaussian
- Orca
- Molpro
- tblite
- A dummy energy surface
"""

# version and other variables
__version__ = '2.1'
__author__ = 'Björn Hein-Janke and Lynn Meeder'

# Easy to import
from .gaussian_interface import gaussian_path_engrads
from .molpro_interface import molpro_path_engrads
from .orca_interface import orca_path_engrads
from .tblite_interface import tblite_path_engrads
from .dummy_interface import dummy_path_engrads
from .engrad_interface import gather_engrfunc

# all
__all__ = ['gaussian_interface', 'molpro_interface', 'orca_interface', 'tblite_interface', 'dummy_interface', 'engrad_interface']