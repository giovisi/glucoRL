"""
Softer reward function for curriculum learning.
Start with this, then switch to the paper's harsh reward after agent learns basics.
"""
import numpy as np

def compute_risk_index(BG):
    """Standard Risk Index calculation"""
    bg_val = max(BG, 1.0)
    f_bg = 1.509 * ((np.log(bg_val) ** 1.084) - 5.381)
    ri = 10 * (f_bg ** 2)
    return ri

def soft_curriculum_reward(bg_hist, **kwargs):
    """
    BALANCED reward function - penalizes hypo AND hyper equally.
    Forces agent to actively manage glucose, not just do nothing.
    """
    g_next = bg_hist[-1]
    
    # Calculate base risk
    raw_ri = compute_risk_index(g_next)
    max_theoretical_risk = 100.0
    normalized_ri = min(raw_ri / max_theoretical_risk, 1.0)
    
    # BALANCED penalties - hyper is now as bad as hypo!
    if g_next <= 39.0:
        # Severe hypo: -5 penalty
        return -5.0 - normalized_ri
    elif g_next < 54.0:
        # Moderate hypo: -2 extra penalty
        return -2.0 - normalized_ri
    elif g_next < 70.0:
        # Mild hypo: -1 extra penalty
        return -1.0 - normalized_ri
    elif g_next > 350.0:
        # Severe hyper: -5 penalty (SAME AS SEVERE HYPO!)
        return -5.0 - normalized_ri
    elif g_next > 250.0:
        # Moderate hyper: -2 penalty (matched to hypo)
        return -2.0 - normalized_ri
    elif g_next > 180.0:
        # Mild hyper: -1 penalty (matched to hypo)
        return -1.0 - normalized_ri
    else:
        # Target range (70-180): just negative RI (best reward near 0)
        return -normalized_ri

def shaped_reward(bg_hist, **kwargs):
    """
    Alternative: Positive reward shaping
    Rewards being in target range, penalizes being out of range.
    """
    g_next = bg_hist[-1]
    
    # Positive reward for being in target range
    if 70 <= g_next <= 180:
        # Best reward when near 100-140
        distance_from_ideal = abs(g_next - 110)
        reward = 1.0 - (distance_from_ideal / 70.0)  # Max reward ~1.0
        return max(reward, 0.1)  # At least 0.1 for being in range
    
    # Graduated penalties outside range
    if g_next < 70:
        if g_next <= 39:
            return -5.0
        elif g_next < 54:
            return -2.0
        else:
            return -1.0
    
    if g_next > 180:
        if g_next > 250:
            return -2.0
        else:
            return -0.5
    
    return 0.0
