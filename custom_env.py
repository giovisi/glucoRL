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

# --- SMART REWARD FUNCTION (Enhanced) ---
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

# --- GYM ENVIRONMENT CLASS ---
class CustomT1DEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None, episode_days=1):
        """
        Initialize the Custom Environment.
        
        Args:
            patient_name: Name of the T1D patient
            custom_scenario: Fixed scenario or None for random generation
            reward_fun: Reward function (paper_reward or smart_reward)
            seed: Random seed
            episode_days: Duration of episode in days (default=1)
        """
        self.patient_name = patient_name
        self.custom_scenario = custom_scenario
        
        # Use the paper's reward function by default
        if reward_fun is None:
            self.reward_fun = paper_reward
        else:
            self.reward_fun = reward_fun
            
        self.seed_val = seed
        self.k_history = 12  # History length: 1 hour (12 * 5 mins)
        self.episode_days = episode_days  # Episode duration in days
        
        # Initialize manual buffers
        self.bg_history_buffer = []
        self.ins_history_buffer = []
        
        # Create internal environment
        self.env, _, _, _ = self._create_env()

        # DYNAMIC: Calculate from sample_time and episode_days
        self.sample_time = self.env.sensor.sample_time  # minutes
        self.max_episode_steps = int((self.episode_days * 24 * 60) / self.sample_time)
        self.current_step = 0
        
        # --- ACTION SPACE ---
        # Single continuous action in [-1, 1] mapped exponentially to insulin rate
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # CRITICAL: Realistic I_max and exponential mapping parameter
        self.I_max = 0.05  # U/min (= 3 U/hr max, realistic)
        self.eta = 4.0     # Exponential mapping steepness

        # --- OBSERVATION SPACE ---
        # State includes past glucose and insulin measurements [cite: 196]
        # plus our added future carb information for the "Smart" agent.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )

    def _generate_random_day_scenario(self):
        """Generate random daily scenario with realistic meal timing and amounts."""
        import random
        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())
        
        meals = []
        # Breakfast: 7-9am, 30-60g
        t_b = start_time + timedelta(hours=7, minutes=random.randint(0, 119))
        meals.append((t_b, random.randint(30, 60)))
        
        # Lunch: 12-2pm, 60-100g
        t_l = start_time + timedelta(hours=12, minutes=random.randint(0, 119))
        meals.append((t_l, random.randint(60, 100)))
        
        # Dinner: 6-8pm, 50-90g
        t_d = start_time + timedelta(hours=18, minutes=random.randint(0, 119))
        meals.append((t_d, random.randint(50, 90)))
        
        meals.sort(key=lambda x: x[0])
        from simglucose.simulation.scenario import CustomScenario
        return CustomScenario(start_time=start_time, scenario=meals)

    def _create_env(self):
        patient = T1DPatient.withName(self.patient_name)
        sensor = CGMSensor.withName('Dexcom', seed=self.seed_val)
        pump = InsulinPump.withName('Insulet')
        
        if self.custom_scenario:
            scenario = self.custom_scenario
        else:
            # Use random daily scenario instead of RandomScenario
            scenario = self._generate_random_day_scenario()
            
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

        # Normalize: BG by 200, insulin by I_max, carbs by 100
        obs = np.concatenate([
            np.array(bg_hist) / 200.0,                     # [0, ~2]
            np.array(ins_hist) / self.I_max,               # [0, ~1] 
            np.array([c_0_30, c_30_60, c_60_120]) / 100.0  # [0, ~2]
        ])
        
        return obs.astype(np.float32)

    def step(self, action):
        """
        Execute one step with exponential action mapping.
        ADDED: Episode termination for catastrophic BG.
        """
        # 1. Increment step counter
        self.current_step += 1
        
        # 2. Clip action to [-1, 1]
        a = np.clip(float(action[0]), -1.0, 1.0)
        
        # 3. Exponential mapping: I_max * exp(eta * (a - 1))
        # This maps [-1, 1] -> [I_max*e^(-eta), I_max] = [~0.0003, 0.05] U/min
        rate_u_min = self.I_max * np.exp(self.eta * (a - 1.0))
        
        # 4. Create action (basal only, no bolus)
        act = Action(basal=rate_u_min, bolus=0)
        
        # 5. Step environment with reward function
        if self.reward_fun:
            obs, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        else:
            obs, reward, done, info = self.env.step(act)
        
        # 6. Initialize termination flags
        terminated = False 
        truncated = False
        
        # 7. Catastrophic failure check
        bg_true = self.env.patient.observation.Gsub
        
        if bg_true <= 39.0 or bg_true >= 600.0:
            terminated = True
            reward = -15.0
            info['catastrophic_failure'] = True
            info['failure_reason'] = f'BG={bg_true:.1f}'
        
        # 8. Update history
        self._update_history(obs.CGM, rate_u_min)
        
        # 9. Get observation
        new_obs = self._get_smart_state()
        
        # 10. Check for episode truncation (max steps reached)
        if self.current_step >= self.max_episode_steps:
            truncated = True
            info['truncated'] = True
        
        return new_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset environment and regenerate scenario if training."""
        self.current_step = 0
        super().reset(seed=seed)
        
        # Regenerate scenario if not using fixed scenario (for training variety)
        if self.custom_scenario is None:
            self.env, _, _, _ = self._create_env()
        
        obs, _, _, _ = self.env.reset()
        
        # Initialize history buffers
        self.bg_history_buffer = [obs.CGM] * self.k_history
        self.ins_history_buffer = [0.0] * self.k_history
        
        new_obs = self._get_smart_state()
        return new_obs, {}
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.render(close=True)