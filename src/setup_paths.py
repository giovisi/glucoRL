"""
Path Configuration Module for the RL-Project.

This module must be imported FIRST in any script that requires access to
the SimGlucose library. It configures the Python path to include the
SimGlucose submodule and resolves common library conflicts.

Usage:
    import setup_paths  # Must be the first import in your script!
    from simglucose.patient.t1dpatient import T1DPatient
    ...

Why is this needed?
- SimGlucose is included as a Git submodule rather than installed via pip
- The submodule path must be added to sys.path before importing SimGlucose
- macOS has a known OpenMP library conflict between NumPy and PyTorch
"""

import os
import sys
import warnings


# =============================================================================
# ENVIRONMENT FIXES
# =============================================================================

# Suppress deprecation warning from simglucose's internal gym import
# (simglucose uses the old OpenAI Gym which triggers a pkg_resources warning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Fix OpenMP duplicate library issue on macOS
# This occurs when both NumPy and PyTorch load their own OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Determine project root directory (parent of the src folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the SimGlucose submodule to Python's module search path
# Using insert(0, ...) ensures it takes priority over any installed versions
SUBMODULE_ROOT = os.path.join(BASE_DIR, "simglucose")
if SUBMODULE_ROOT not in sys.path:
    sys.path.insert(0, SUBMODULE_ROOT)
