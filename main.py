import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import os

# Import the helpers module 
import helpers

# --- Simglucose Imports and Explanations ---

# BBController: Implements the standard "Basal-Bolus" therapy.
# It delivers a constant background insulin (basal) and calculates large doses (bolus) 
# for meals based on Carb Ratio (CR) and Correction Factor (CF). We will use this as baseline for comparisons
from simglucose.controller.basal_bolus_ctrller import BBController

# PIDController: Implements a Proportional-Integral-Derivative control algorithm.
# It calculates insulin delivery continuously based on the error between 
# current glucose and the target glucose.
from simglucose.controller.pid_ctrller import PIDController

# T1DPatient: Loads a virtual patient model based on the FDA-approved UVA/Padova simulator.
# It contains the differential equations describing how the patient's body metabolizes glucose and insulin.
from simglucose.patient.t1dpatient import T1DPatient

# CGMSensor: Simulates a Continuous Glucose Monitor.
# It reads the "true" blood glucose from the patient and adds realistic sensor noise 
# and time lag before passing it to the controller.
from simglucose.sensor.cgm import CGMSensor

# InsulinPump: Simulates the hardware device.
# It receives commands from the controller and ensures insulin delivery limits 
# and discrete step sizes are respected.
from simglucose.actuator.pump import InsulinPump

# T1DSimEnv: The "Environment" that connects the Patient, Sensor, and Pump.
# It follows the OpenAI Gym standard (commonly used in Reinforcement Learning),
# providing methods like .step() to advance time.
from simglucose.simulation.env import T1DSimEnv

# CustomScenario: Defines the "world events" for the simulation.
# Specifically, it sets the time and amount of carbohydrate intake (meals).
from simglucose.simulation.scenario import CustomScenario

# SimObj & sim: The simulation engine.
# SimObj wraps the environment and the controller together.
# sim() is the function that actually runs the loop (time step by time step).
from simglucose.simulation.sim_engine import SimObj, sim

# --- 1. CONFIGURATION ---
SIM_DAYS = 1
PATIENT_ID = 'adolescent#003'

# --- 2. SETUP FUNCTION ---
# This function creates a fresh environment instance for every simulation.
# It is crucial to do this because the Patient object holds "state" (current glucose, 
# active insulin, etc.). If we reuse the same environment for the second controller, 
# the patient would already be altered by the first simulation.
def create_env():
    # 1. Create the Patient
    # Loads mathematical parameters for specific virtual patients (e.g., 'adolescent#003')
    patient = T1DPatient.withName(PATIENT_ID)
    
    # 2. Create the Sensor
    # 'Dexcom' is a preset noise model. 'seed' ensures the noise is reproducible.
    sensor = CGMSensor.withName('Dexcom', seed=1)
    
    # 3. Create the Pump
    # 'Insulet' is a preset pump model (Omnipod) with specific delivery constraints.
    pump = InsulinPump.withName('Insulet')
    
    # 4. Define the Meal Scenario (Breakfast, Lunch, Dinner)

    # datetime.now() gets the current local date and time from your computer's clock.
    now = datetime.now()

    # We want the simulation to start at midnight of the current day.
    # .date() extracts just the date (YYYY-MM-DD) from 'now'.
    # .min.time() gives the earliest possible time (00:00:00).
    # .combine() merges them to create a specific timestamp: Today at Midnight.
    start_time = datetime.combine(now.date(), datetime.min.time())

    # We use timedelta to define durations (time differences) to schedule meals relative to the start time.
    meal_events = [
        # start_time + 30 minutes duration = Breakfast time
        (start_time + timedelta(minutes=30), 50),
        
        # start_time + 4 hours duration = Lunch time
        (start_time + timedelta(hours=4), 70),
        
        # start_time + 10 hours duration = Dinner time
        (start_time + timedelta(hours=10), 60)
    ]
    
    # CustomScenario creates the schedule based on the list above
    meal_scenario = CustomScenario(start_time=start_time, scenario=meal_events)
    

    # 5. Combine everything into the Environment
    return T1DSimEnv(patient, sensor, pump, meal_scenario)




def main():
    print(f"--- Starting Simulation for {PATIENT_ID} ---")

    # --- 3. CONTROLLER CONFIGURATION ---
    
    # Controller 1: Basal Bolus
    # This uses standard clinical logic (reactive to meals).
    bb_controller = BBController()

    # Controller 2: PID
    # This uses control theory logic (reactive to glucose deviation).
    # P (Proportional): Reacts to current error (Target - Current BG).
    # I (Integral): Reacts to accumulation of past errors (handles steady drift).
    # D (Derivative): Reacts to the rate of change (predicts future trend).
    # These parameters are generic; optimal values vary per patient.
    pid_controller = PIDController(
        P=1.00e-04, 
        I=1.00e-07, 
        D=3.98e-03, 
        target=140
    )


    from simglucose.simulation.user_interface import simulate
    from simglucose.controller.base import Controller, Action

    class rlController(Controller):
        def __init__(self, init_state, target_bg=140, sensitivity=50):
            """
            init_state: Initial state (usually 0 or None)
            target_bg: The glucose level we want to maintain (default 140 mg/dL)
            sensitivity: How much glucose is lowered by 1 unit of insulin (ISF).
                         A rough estimate for an adolescent is ~50.
            """
            self.init_state = init_state
            self.state = init_state
            self.target_bg = target_bg
            self.sensitivity = sensitivity
            # A standard basal rate estimate for an adolescent (approx 0.05 U/min)
            self.constant_basal = 0.05 

        def policy(self, observation, reward, done, **info):
            '''
            A simple heuristic policy:
            If Glucose > Target, inject corrective bolus.
            Always inject a constant basal rate.
            '''
            self.state = observation
            
            # 1. Get current glucose
            bg_current = observation.CGM

            # 2. Calculate "Correction Bolus"
            # Formula: (Current - Target) / Sensitivity
            if bg_current > self.target_bg:
                # Example: If BG is 200, Target 140, Sens 50.
                # Diff = 60. Bolus = 60/50 = 1.2 Units.
                calculated_bolus = (bg_current - self.target_bg) / self.sensitivity
            else:
                calculated_bolus = 0

            # 3. Safety: Ensure we don't deliver negative values or crazy high amounts
            # Limiting to max 5 units per step to prevent death in simulation
            safe_bolus = min(max(calculated_bolus, 0), 5.0)

            # 4. Return Action
            # Basal keeps the patient stable; Bolus corrects the highs.
            return Action(basal=self.constant_basal, bolus=safe_bolus)

        def reset(self):
            '''
            Reset the controller state to initial state
            '''
            self.state = self.init_state

    rl_controller = rlController(0)

    # --- 4. EXECUTE SIMULATIONS ---
    
    # --- Simulation A: Basal Bolus ---
    print("\nRunning Basal-Bolus Simulation...")
    env_bb = create_env() # Create fresh environment
    
    # SimObj binds the environment (patient) to the controller (brain)
    sim_obj_bb = SimObj(
        env_bb, 
        bb_controller, 
        timedelta(days=SIM_DAYS), 
        animate=False, 
        path='./temp_results_bb' # Temporary path for internal logs
    )
    # The sim() function executes the Ordinary Differential Equations (ODEs)
    # to calculate glucose changes over time.
    results_bb = sim(sim_obj_bb)

    # --- Simulation B: PID ---
    print("\nRunning PID Simulation...")
    env_pid = create_env() # Create fresh environment
    
    sim_obj_pid = SimObj(
        env_pid, 
        pid_controller, 
        timedelta(days=SIM_DAYS), 
        animate=False, 
        path='./temp_results_pid'
    )
    results_pid = sim(sim_obj_pid)

     # --- Simulation C: RL ---
    print("\nRunning RL Simulation...")
    env_rl = create_env() # Create fresh environment
    
    sim_obj_rl = SimObj(
        env_rl,
        rl_controller, 
        timedelta(days=SIM_DAYS), 
        animate=False, 
        path='./temp_results_rl'
    )
    results_rl = sim(sim_obj_rl)


    # --- 5. COLLECT RESULTS ---
    # Store dataframes in a dictionary for easier processing by helper functions
    results_dict = {
        'Basal-Bolus': results_bb,
        'PID': results_pid,
        'RL' : results_rl
    }

    # --- 6. DISPLAY METRICS ---
    print("\n" + "="*30)
    print("      METRIC RESULTS      ")
    print("="*30)
    
    # Calculate and print stats (Time in Range, Mean BG, etc.)
    helpers.print_glycemic_metrics(results_bb, "Basal-Bolus")
    helpers.print_glycemic_metrics(results_pid, "PID")
    helpers.print_glycemic_metrics(results_rl,"RL")

    # --- 7. SAVE FILES AND PLOTS ---
    # Create a unique folder name based on the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"Simulation_{PATIENT_ID}_{timestamp}"
    
    print(f"\nSaving results to folder: {folder_name}...")
    
    # Call the helper function to save CSV files and generate plots
    helpers.save_all_results(results_dict, folder_name=folder_name)
    
    print("\nSimulation completed successfully.")

if __name__ == "__main__":
    main()