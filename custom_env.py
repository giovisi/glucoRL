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

    def __init__(self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None, max_episode_days=1, allow_early_termination=False):
        """
        Initialize the Custom Environment.
        
        Args:
            allow_early_termination: If False, episodes run for full max_episode_days
                                   If True, episodes can end on dangerous glucose
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
        
        # Episode management
        self.max_episode_steps = int(max_episode_days * 288)  # 288 steps = 1 day
        self.allow_early_termination = allow_early_termination
        self.current_step = 0
        
        # Initialize manual buffers
        self.bg_history_buffer = []
        self.ins_history_buffer = []
        
        # Create internal environment
        self.env, _, _, _ = self._create_env()

        # --- ACTION SPACE ---
        # Scale actions to realistic insulin ranges
        # REDUCED to prevent overdosing during learning
        # Basal: 0-0.15 U/min (more conservative)
        # Bolus: 0-2.5 U (smaller meal bolus)
        self.max_basal = 0.15  # U/min (reduced from 0.3)
        self.max_bolus = 2.5   # U (reduced from 5.0)
        
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

        # Better normalization:
        # BG: Divide by 400 to keep most values in [0, 1]
        # Insulin: Divide by max possible insulin per 5 min (0.3 * 5 + 5 = 6.5)
        # Carbs: Divide by typical meal size (75g)
        obs = np.concatenate([
            np.array(bg_hist) / 400.0,  # BG normalization
            np.array(ins_hist) / 10.0,  # Insulin normalization
            np.array([c_0_30, c_30_60, c_60_120]) / 75.0  # Carb normalization
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
        
        # Get current glucose (CGM reading)
        current_bg = obs.CGM if obs.CGM is not None else self.env.patient.observation.Gsub
        
        # Increment step counter
        self.current_step += 1
        
        # Control termination: allow agent to learn recovery unless configured otherwise
        if not self.allow_early_termination:
            # Override done - only end after max steps (let agent learn from mistakes)
            done = self.current_step >= self.max_episode_steps
        else:
            # Keep environment's natural termination OR max steps
            done = done or (self.current_step >= self.max_episode_steps)
        
        # Update history and get state
        self._update_history(current_bg, basal_act + bolus_act)
        new_obs = self._get_smart_state()
        
        # Enhanced info dict for debugging and monitoring
        info['glucose'] = current_bg
        info['basal'] = basal_act
        info['bolus'] = bolus_act
        info['total_insulin'] = basal_act + bolus_act
        info['episode_step'] = self.current_step
        
        return new_obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _, _, _ = self.env.reset()
        
        # Reset step counter
        self.current_step = 0
        
        # Get initial glucose
        initial_bg = obs.CGM if obs.CGM is not None else self.env.patient.observation.Gsub
        
        self.bg_history_buffer = [initial_bg] * self.k_history
        self.ins_history_buffer = [0.0] * self.k_history
        
        new_obs = self._get_smart_state()
        return new_obs, {}
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.render(close=True)