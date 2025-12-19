import numpy as np

# --- RISK INDEX CALCULATION (Standard Formula) ---
def compute_risk_index(BG):
    """
    Calculates the standard Risk Index (RI) based on Kovatchev et al.[cite: 208].
    This computes the raw risk value before any normalization.
    """
    # Ensure BG is at least 1.0 to avoid mathematical errors with log
    bg_val = max(BG, 1.0) 
    
    # 1. Apply the standard logarithmic transformation function f(BG)
    # Constants 1.509, 1.084, and 5.381 are standard parameters for this medical formula
    f_bg = 1.509 * ((np.log(bg_val) ** 1.084) - 5.381)
    
    # 2. Calculate the Risk Index: 10 * f(BG)^2
    ri = 10 * (f_bg ** 2)
    
    return ri

# --- REWARD FUNCTION (G2P2C Paper Implementation) ---
def paper_reward(bg_hist, **kwargs):
    """
    Implements the exact reward function defined in Equation (6) of the G2P2C paper[cite: 211, 214].
    
    Formula:
    R(s,a) = -15                 if g_{t+1} <= 39 mg/dL (Severe Hypoglycemia)
    R(s,a) = -1 * Normalized_RI  otherwise
    """
    # Get the most recent glucose value (g_{t+1})
    g_next = bg_hist[-1]
    
    # --- Condition 1: Severe Hypoglycemia ---
    # The paper specifies a hard penalty for severe hypoglycemia[cite: 212].
    if g_next <= 39.0:
        return -15.0
        
    # --- Condition 2: Normalized Risk Index ---
    # For the rest of the range, the paper uses a negative normalized RI[cite: 213].
    # Normalization implies scaling the RI to the range [0, 1].
    # Theoretical maximum risk is often considered around 100.
    raw_ri = compute_risk_index(g_next)
    max_theoretical_risk = 100.0 
    normalized_ri = min(raw_ri / max_theoretical_risk, 1.0)
    
    # Return the negative normalized risk (Max Reward = 0, Min Reward = -1)
    return -1.0 * normalized_ri

# --- SMART REWARD FUNCTION (Enhanced) ---
# Note: this function is not used by default, but provided for experimentation.
def smart_reward(bg_hist, **kwargs):
    """
    Enhanced reward function with positive rewards in target range.
    More aggressive penalties for severe conditions.
    """
    bg = bg_hist[-1]
    
    # Catastrophic hypoglycemia
    if bg <= 40:
        return -100.0
    
    # Target range: positive reward
    if 70 <= bg <= 150:
        return 1.0
    
    # Acceptable range: smaller positive reward
    if 150 < bg <= 180:
        return 0.5
    
    # Out of range: negative reward based on risk
    risk = compute_risk_index(bg)
    multiplier = 2.0 if bg > 250 else 1.0
    return -1.0 * (risk / 50.0) * multiplier