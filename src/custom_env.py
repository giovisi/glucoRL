"""
Custom Gymnasium Environment for Type 1 Diabetes (T1D) Blood Glucose Control.

This module implements a reinforcement learning environment for training agents
to control blood glucose levels in Type 1 Diabetic virtual patients using
the simglucose simulator.

The environment uses:
- Continuous action space for insulin delivery (exponentially mapped)
- Observation space with glucose/insulin history + future carb info ("Smart" agent)
- Customizable reward functions for training different control strategies
"""

import gymnasium as gym
import numpy as np
from datetime import datetime, timedelta

# SimGlucose components for T1D simulation
from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action

from reward_functions import paper_reward


class CustomT1DEnv(gym.Env):
    """
    Custom Gymnasium environment wrapping the SimGlucose T1D simulator.
    
    This environment provides a "Smart" agent setup where the observation
    includes historical glucose/insulin data plus future meal (carbohydrate)
    information, allowing the agent to anticipate and preemptively dose insulin.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None, episode_days=1):
        """
        Initialise the Custom Environment.
        
        Args:
            patient_name: Name of the T1D patient
            custom_scenario: Fixed scenario or None for random generation
            reward_fun: Reward function (default: paper_reward)
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
        
        # History length for observations: 12 steps = 1 hour (12 * 5-minute intervals)
        # This provides the agent with recent glucose and insulin trends
        self.k_history = 12
        self.episode_days = episode_days  # Episode duration in days
        
        # Rolling buffers to store recent blood glucose and insulin values
        # These are used to construct the observation state
        self.bg_history_buffer = []
        self.ins_history_buffer = []
        
        # Create internal environment
        self.env, _, _, _ = self._create_env()

        # Calculate episode length dynamically based on sensor sample time
        # Sample time is typically 5 minutes for CGM sensors
        self.sample_time = self.env.sensor.sample_time  # minutes per step
        self.max_episode_steps = int((self.episode_days * 24 * 60) / self.sample_time)
        self.current_step = 0
        
        # ==================== ACTION SPACE ====================
        # Continuous action in [-1, 1], later mapped exponentially to insulin rate.
        # This allows the RL agent to output normalized actions that we then
        # transform into physiologically meaningful insulin delivery rates.
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Insulin delivery parameters for exponential action mapping:
        # - I_max: Maximum insulin rate (0.05 U/min = 3 U/hr) - clinically realistic
        # - eta: Steepness of exponential curve (higher = more sensitive near max)
        # Mapping formula: insulin = I_max * exp(eta * (action - 1))
        # This maps action=-1 to ~0.0003 U/min, action=1 to 0.05 U/min
        self.I_max = 0.05  # U/min (= 3 U/hr max, realistic for basal delivery)
        self.eta = 4.0     # Exponential mapping steepness parameter

        # ==================== OBSERVATION SPACE ====================
        # "Smart" agent observation includes:
        # - 12 past BG readings (1 hour of history, normalized by 200 mg/dL)
        # - 12 past insulin rates (1 hour of history, normalized by I_max)
        # - 3 future carb windows: [0-30min, 30-60min, 60-120min] (normalized by 100g)
        # Total: 12 + 12 + 3 = 27 features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )

    def _generate_random_day_scenario(self):
        """
        Generate a random daily meal scenario with realistic timing and amounts.
        
        Creates three meals (breakfast, lunch, dinner) with randomized:
        - Timing: within typical meal windows
        - Carbohydrate amounts: within realistic ranges for each meal type
        
        Returns:
            CustomScenario: A simglucose scenario object with the meal schedule
        """
        import random
        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())  # Midnight
        
        meals = []
        
        # Breakfast: between 7:00-9:00 AM, moderate carbs (30-60g)
        t_b = start_time + timedelta(hours=7, minutes=random.randint(0, 119))
        meals.append((t_b, random.randint(30, 60)))
        
        # Lunch: between 12:00-2:00 PM, larger meal (60-100g)
        t_l = start_time + timedelta(hours=12, minutes=random.randint(0, 119))
        meals.append((t_l, random.randint(60, 100)))
        
        # Dinner: between 6:00-8:00 PM, medium-large meal (50-90g)
        t_d = start_time + timedelta(hours=18, minutes=random.randint(0, 119))
        meals.append((t_d, random.randint(50, 90)))
        
        # Sort meals chronologically (should already be in order, but ensures correctness)
        meals.sort(key=lambda x: x[0])
        
        from simglucose.simulation.scenario import CustomScenario
        return CustomScenario(start_time=start_time, scenario=meals)

    def _create_env(self):
        """
        Create and configure the internal SimGlucose T1D simulation environment.
        
        Initializes:
        - Virtual patient with T1D physiology
        - Dexcom CGM sensor for glucose readings
        - Insulet insulin pump for delivery
        - Meal scenario (custom or randomly generated)
        
        Returns:
            tuple: (env, 0, 0, 0) - environment and placeholder values for compatibility
        """
        # Create virtual T1D patient from the SimGlucose patient database
        patient = T1DPatient.withName(self.patient_name)
        
        # Configure Dexcom CGM sensor (provides glucose readings every 5 min)
        sensor = CGMSensor.withName('Dexcom', seed=self.seed_val)
        
        # Configure Insulet insulin pump for continuous insulin delivery
        pump = InsulinPump.withName('Insulet')
        
        # Use provided scenario or generate random daily meals for training variety
        if self.custom_scenario:
            scenario = self.custom_scenario
        else:
            scenario = self._generate_random_day_scenario()
            
        # Assemble the complete T1D simulation environment
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, 0, 0, 0

    def _sum_carbs_in_window(self, t_start, t_end):
        """
        Calculate total carbohydrates from meals within a specified time window.
        
        This is used for the "Smart" agent's lookahead feature, allowing
        the agent to anticipate upcoming meals and adjust insulin accordingly.
        
        Args:
            t_start: Start of the time window (datetime)
            t_end: End of the time window (datetime)
            
        Returns:
            float: Total grams of carbohydrates in the window
        """
        amt = 0
        if hasattr(self.env.scenario, 'scenario'):
            # RandomScenario doesn't expose future meals, so return 0
            if isinstance(self.env.scenario, RandomScenario):
                return 0 
            else:
                # Iterate through scheduled meals and sum carbs in window
                for meal_time, meal_val in self.env.scenario.scenario:
                    if t_start <= meal_time < t_end:
                        amt += meal_val
        return amt

    def _update_history(self, bg, insulin):
        """
        Update the rolling history buffers with new glucose and insulin values.
        
        Maintains fixed-length buffers (k_history) by discarding oldest values
        when the buffer exceeds the maximum length.
        
        Args:
            bg: Current blood glucose reading (mg/dL)
            insulin: Current insulin delivery rate (U/min)
        """
        self.bg_history_buffer.append(bg)
        self.ins_history_buffer.append(insulin)
        
        # Keep only the most recent k_history values (sliding window)
        if len(self.bg_history_buffer) > self.k_history:
            self.bg_history_buffer = self.bg_history_buffer[-self.k_history:]
        if len(self.ins_history_buffer) > self.k_history:
            self.ins_history_buffer = self.ins_history_buffer[-self.k_history:]

    def _get_smart_state(self):
        """
        Construct the "Smart" agent observation state.
        
        The observation includes:
        1. Blood glucose history (past 1 hour, 12 values)
        2. Insulin delivery history (past 1 hour, 12 values)  
        3. Future carbohydrate information (3 time windows)
        
        All values are normalized to roughly [0, 1-2] range for stable training.
        
        Returns:
            np.ndarray: 27-dimensional observation vector (float32)
        """
        # Pad glucose history if not enough data (e.g., at episode start)
        bg_hist = list(self.bg_history_buffer)
        if len(bg_hist) < self.k_history:
            missing = self.k_history - len(bg_hist)
            val = bg_hist[-1] if len(bg_hist) > 0 else 140.0  # Default to normal BG
            bg_hist = [val] * missing + bg_hist
        
        # Pad insulin history similarly
        ins_hist = list(self.ins_history_buffer)
        if len(ins_hist) < self.k_history:
            missing = self.k_history - len(ins_hist)
            val = ins_hist[-1] if len(ins_hist) > 0 else 0.0
            ins_hist = [val] * missing + ins_hist

        # Lookahead: get upcoming carbohydrate intake in three time windows
        # This is the "Smart" feature that distinguishes this agent
        t_now = self.env.time
        c_0_30 = self._sum_carbs_in_window(t_now, t_now + timedelta(minutes=30))
        c_30_60 = self._sum_carbs_in_window(t_now + timedelta(minutes=30), t_now + timedelta(minutes=60))
        c_60_120 = self._sum_carbs_in_window(t_now + timedelta(minutes=60), t_now + timedelta(minutes=120))

        # Normalize all features to similar scales for neural network training:
        # - BG: divide by 200 (normal ~0.5, high ~1.5)
        # - Insulin: divide by I_max (range [0, 1])
        # - Carbs: divide by 100g (typical meal ~0.5-1.0)
        obs = np.concatenate([
            np.array(bg_hist) / 200.0,                     # 12 values: BG history
            np.array(ins_hist) / self.I_max,               # 12 values: insulin history
            np.array([c_0_30, c_30_60, c_60_120]) / 100.0  # 3 values: future carbs
        ])
        
        return obs.astype(np.float32)


    def step(self, action):
        """
        Execute one environment step: apply insulin action and observe result.
        
        This method:
        1. Maps the normalized action to a physical insulin rate
        2. Steps the simulation forward by one time step (5 minutes)
        3. Computes reward based on resulting blood glucose
        4. Checks for catastrophic failures (severe hypo/hyperglycemia)
        
        Args:
            action: np.ndarray of shape (1,) with value in [-1, 1]
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Step 1: Increment step counter for episode length tracking
        self.current_step += 1
        
        # Step 2: Clip action to valid range [-1, 1] (safety measure)
        a = np.clip(float(action[0]), -1.0, 1.0)
        
        # Step 3: Exponential action mapping for insulin rate
        # Formula: I_max * exp(eta * (a - 1))
        # This creates a non-linear mapping where:
        #   - action = -1 → ~0.0003 U/min (minimal insulin)
        #   - action =  0 → ~0.001 U/min (low insulin)
        #   - action =  1 → 0.05 U/min (maximum insulin)
        rate_u_min = self.I_max * np.exp(self.eta * (a - 1.0))
        
        # Step 4: Create SimGlucose action (continuous basal, no meal bolus)
        act = Action(basal=rate_u_min, bolus=0)
        
        # Step 5: Execute action in simulation and get results
        if self.reward_fun:
            obs, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        else:
            obs, reward, done, info = self.env.step(act)
        
        # Step 6: Initialize Gymnasium termination flags
        terminated = False  # True if episode ends due to failure
        truncated = False   # True if episode ends due to time limit
        
        # Step 7: Check for catastrophic blood glucose levels
        # Severe hypoglycemia (<39 mg/dL) or severe hyperglycemia (>600 mg/dL)
        # These are life-threatening conditions that end the episode
        bg_true = self.env.patient.observation.Gsub  # True BG (not CGM reading)
        
        if bg_true <= 39.0 or bg_true >= 600.0:
            # Catastrophic failure: apply large negative reward and terminate
            terminated = True
            reward = -15.0  # Strong penalty to discourage dangerous control
            info['catastrophic_failure'] = True
            info['failure_reason'] = f'BG={bg_true:.1f}'
        
        # Step 8: Update rolling history buffers with new readings
        self._update_history(obs.CGM, rate_u_min)
        
        # Step 9: Construct observation for the agent
        new_obs = self._get_smart_state()
        
        # Step 10: Check for episode truncation (reached max episode length)
        if self.current_step >= self.max_episode_steps:
            truncated = True
            info['truncated'] = True
        
        return new_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        If no custom scenario is set, generates a new random meal scenario
        for each episode to provide training variety.
        
        Args:
            seed: Random seed for reproducibility (optional)
            options: Additional reset options (unused, for Gymnasium compatibility)
            
        Returns:
            tuple: (initial_observation, info_dict)
        """
        self.current_step = 0
        super().reset(seed=seed)
        
        # Generate new random meal scenario for training variety
        # (skip if using a fixed custom scenario for evaluation)
        if self.custom_scenario is None:
            self.env, _, _, _ = self._create_env()
        
        # Reset the simulation and get initial observation
        obs, _, _, _ = self.env.reset()
        
        # Initialise history buffers with initial CGM reading
        # (pad with same value to provide full history from start)
        self.bg_history_buffer = [obs.CGM] * self.k_history
        self.ins_history_buffer = [0.0] * self.k_history  # No insulin delivered yet
        
        # Construct and return the initial observation state
        new_obs = self._get_smart_state()
        return new_obs, {}
    
    def render(self):
        """Render the current environment state (visualization)."""
        self.env.render()
    
    def close(self):
        """Clean up resources and close any open render windows."""
        self.env.render(close=True)