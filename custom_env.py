import gymnasium as gym
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action

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

# --- GYM ENVIRONMENT CLASS ---
class CustomT1DEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None):
        """
        Initialize the Custom Environment.
        """
        self.patient_name = patient_name
        self.custom_scenario = custom_scenario
        
        # Use the paper's reward function by default
        if reward_fun is None:
            self.reward_fun = paper_reward
        else:
            self.reward_fun = reward_fun
            
        self.seed_val = seed
        self.k_history = 12  # History length: 1 hour (12 * 5 mins) [cite: 196]
        
        # Initialize manual buffers
        self.bg_history_buffer = []
        self.ins_history_buffer = []
        
        # Create internal environment
        self.env, _, _, _ = self._create_env()

        # --- ACTION SPACE ---
        # The paper uses a continuous action space mapped to [0, 5] U/min[cite: 202].
        self.max_basal = self.env.pump._params['max_basal']
        self.max_bolus = 5.0 
        
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0]), 
            high=np.array([self.max_basal, self.max_bolus]), 
            dtype=np.float32
        )

        # --- OBSERVATION SPACE ---
        # State includes past glucose and insulin measurements [cite: 196]
        # plus our added future carb information for the "Smart" agent.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )

    def _create_env(self):
        patient = T1DPatient.withName(self.patient_name)
        sensor = CGMSensor.withName('Dexcom', seed=self.seed_val)
        pump = InsulinPump.withName('Insulet')
        
        if self.custom_scenario:
            scenario = self.custom_scenario
        else:
            start_time = datetime(2018, 1, 1, 0, 0, 0)
            scenario = RandomScenario(start_time=start_time, seed=self.seed_val)
            
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, 0, 0, 0

    def _sum_carbs_in_window(self, t_start, t_end):
        amt = 0
        if hasattr(self.env.scenario, 'scenario'):
             if isinstance(self.env.scenario, RandomScenario):
                 return 0 
             else:
                 for meal_time, meal_val in self.env.scenario.scenario:
                     if t_start <= meal_time < t_end:
                         amt += meal_val
        return amt

    def _update_history(self, bg, insulin):
        self.bg_history_buffer.append(bg)
        self.ins_history_buffer.append(insulin)
        
        if len(self.bg_history_buffer) > self.k_history:
            self.bg_history_buffer = self.bg_history_buffer[-self.k_history:]
        if len(self.ins_history_buffer) > self.k_history:
            self.ins_history_buffer = self.ins_history_buffer[-self.k_history:]

    def _get_smart_state(self):
        bg_hist = list(self.bg_history_buffer)
        if len(bg_hist) < self.k_history:
            missing = self.k_history - len(bg_hist)
            val = bg_hist[-1] if len(bg_hist) > 0 else 140.0
            bg_hist = [val] * missing + bg_hist
            
        ins_hist = list(self.ins_history_buffer)
        if len(ins_hist) < self.k_history:
            missing = self.k_history - len(ins_hist)
            val = ins_hist[-1] if len(ins_hist) > 0 else 0.0
            ins_hist = [val] * missing + ins_hist

        t_now = self.env.time
        c_0_30 = self._sum_carbs_in_window(t_now, t_now + timedelta(minutes=30))
        c_30_60 = self._sum_carbs_in_window(t_now + timedelta(minutes=30), t_now + timedelta(minutes=60))
        c_60_120 = self._sum_carbs_in_window(t_now + timedelta(minutes=60), t_now + timedelta(minutes=120))

        obs = np.concatenate([
            np.array(bg_hist) / 200.0,
            np.array(ins_hist) * 1.0, 
            np.array([c_0_30, c_30_60, c_60_120]) / 50.0 
        ])
        
        return obs.astype(np.float32)

    def step(self, action):
        basal_act = float(action[0])
        bolus_act = float(action[1])
        act = Action(basal=basal_act, bolus=bolus_act)
        
        # Step using the paper's reward function logic
        if self.reward_fun:
            obs, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        else:
            obs, reward, done, info = self.env.step(act)
        
        # Update history and get state
        self._update_history(obs.CGM, basal_act + bolus_act)
        new_obs = self._get_smart_state()
        
        
        return new_obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _, _, _ = self.env.reset()
        
        self.bg_history_buffer = [obs.CGM] * self.k_history
        self.ins_history_buffer = [0.0] * self.k_history
        
        new_obs = self._get_smart_state()
        return new_obs, {}
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.render(close=True)