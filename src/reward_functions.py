"""
Reward Functions for Reinforcement Learning Blood Glucose Control.

This module implements reward functions that shape the RL agent's behaviour
during training. The primary function follows the G2P2C paper formulation,
using a risk-based approach to penalise dangerous glucose levels.

The reward signal guides the agent to:
- Maintain glucose within the target range (70-180 mg/dL)
- Avoid hypoglycaemia (low BG) more aggressively than hyperglycaemia
- Apply maximum penalty for severe hypoglycaemia (<39 mg/dL)
"""

import numpy as np


# =============================================================================
# RISK INDEX CALCULATION
# =============================================================================

def compute_risk_index(BG):
    """
    Calculate the standard Blood Glucose Risk Index (RI).
    
    Based on Kovatchev et al., this metric quantifies the clinical risk
    associated with a given blood glucose level. The function is asymmetric,
    assigning higher risk to hypoglycaemia than equivalent hyperglycaemia.
    
    The logarithmic transformation maps the BG scale to a symmetric risk
    space where both low and high values deviate from the optimal range.
    
    Args:
        BG: Blood glucose value in mg/dL
        
    Returns:
        float: Risk index value (higher = more dangerous)
    """
    # Clamp BG to minimum of 1.0 to avoid log(0) mathematical errors
    bg_val = max(BG, 1.0) 
    
    # Apply the standard logarithmic transformation function f(BG)
    # Constants 1.509, 1.084 and 5.381 are clinically validated parameters
    f_bg = 1.509 * ((np.log(bg_val) ** 1.084) - 5.381)
    
    # Calculate the Risk Index: 10 * f(BG)^2
    # Squaring ensures positive values and amplifies extreme deviations
    ri = 10 * (f_bg ** 2)
    
    return ri

# =============================================================================
# PRIMARY REWARD FUNCTION (G2P2C Paper)
# =============================================================================

def paper_reward(bg_hist, **kwargs):
    """
    Reward function based on the G2P2C paper (Equation 6).
    
    This is the default reward function used during training. It provides:
    - Large negative reward (-15) for severe hypoglycaemia
    - Normalised negative risk for all other glucose values
    - Maximum reward of 0 when glucose is in optimal range
    
    The asymmetric risk index naturally penalises hypoglycaemia more
    heavily than hyperglycaemia, reflecting clinical priorities.
    
    Args:
        bg_hist: List/array of blood glucose history, most recent value last
        **kwargs: Additional arguments (unused, for interface compatibility)
        
    Returns:
        float: Reward value in range [-15, 0]
    """
    # Extract the most recent glucose value (the result of the action)
    g_next = bg_hist[-1]
    
    # -------------------------------------------------------------------------
    # Condition 1: Severe Hypoglycaemia (BG <= 39 mg/dL)
    # Apply maximum penalty - this is a life-threatening condition
    # -------------------------------------------------------------------------
    if g_next <= 39.0:
        return -15.0
        
    # -------------------------------------------------------------------------
    # Condition 2: Normal Range - Use Normalised Risk Index
    # Scale the risk to [0, 1] range for stable training gradients
    # -------------------------------------------------------------------------
    raw_ri = compute_risk_index(g_next)
    
    # Theoretical maximum risk (~100) used for normalisation
    max_theoretical_risk = 100.0 
    normalised_ri = min(raw_ri / max_theoretical_risk, 1.0)
    
    # Return negative normalised risk: optimal BG gives ~0, risky BG gives ~-1
    return -1.0 * normalised_ri


# =============================================================================
# EXPERIMENTAL REWARD FUNCTION
# =============================================================================

def smart_reward(bg_hist, **kwargs):
    """
    Enhanced reward function with explicit positive rewards in target range.
    
    NOTE: This function is not used by default. It is provided for
    experimentation with alternative reward shaping strategies.
    
    Key differences from paper_reward:
    - Provides positive rewards (+1.0) when BG is in optimal range
    - Uses tiered positive rewards for acceptable vs optimal ranges
    - Applies stronger penalties for severe hyperglycaemia (>250 mg/dL)
    - Much larger penalty (-100) for catastrophic hypoglycaemia
    
    Args:
        bg_hist: List/array of blood glucose history, most recent value last
        **kwargs: Additional arguments (unused, for interface compatibility)
        
    Returns:
        float: Reward value (positive in target, negative outside)
    """
    bg = bg_hist[-1]
    
    # Catastrophic hypoglycaemia: maximum penalty
    if bg <= 40:
        return -100.0
    
    # Optimal target range (70-150 mg/dL): full positive reward
    if 70 <= bg <= 150:
        return 1.0
    
    # Acceptable range (150-180 mg/dL): partial positive reward
    if 150 < bg <= 180:
        return 0.5
    
    # Out of range: negative reward scaled by risk index
    # Apply 2x multiplier for severe hyperglycaemia to discourage it
    risk = compute_risk_index(bg)
    multiplier = 2.0 if bg > 250 else 1.0
    return -1.0 * (risk / 50.0) * multiplier