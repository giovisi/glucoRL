"""
Path setup for the RL-Project.
Import this module FIRST in any script that needs simglucose.

Usage:
    import setup_paths  # Must be first import!
    from simglucose.patient.t1dpatient import T1DPatient
    ...
"""
import os
import sys

# Project root = folder where this file lives
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add the simglucose submodule to sys.path (only once)
SUBMODULE_ROOT = os.path.join(PROJECT_ROOT, "simglucose")
if SUBMODULE_ROOT not in sys.path:
    sys.path.insert(0, SUBMODULE_ROOT)
