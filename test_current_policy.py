import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMODULE_ROOT = os.path.join(PROJECT_ROOT, "simglucose")
sys.path.insert(0, SUBMODULE_ROOT)

from stable_baselines3 import PPO
from custom_env import CustomT1DEnv
from curriculum_reward import soft_curriculum_reward

print("="*70)
print("TESTING CURRENT POLICY BEHAVIOR")
print("="*70)

# Try SAC model first, fall back to PPO
from stable_baselines3 import SAC
from train_sac import simple_error_reward

sac_model_path = "./train/checkpoints_sac/best_model"
ppo_model_path = "./train/checkpoints_paper/best_model"

if os.path.exists(f"{sac_model_path}.zip"):
    print("Loading SAC model...")
    model = SAC.load(sac_model_path)
    reward_fun = simple_error_reward
elif os.path.exists(f"{ppo_model_path}.zip"):
    print("Loading PPO model...")
    model = PPO.load(ppo_model_path)
    reward_fun = soft_curriculum_reward
else:
    print("No model found! Train first.")
    sys.exit(1)

# Create test environment
env = CustomT1DEnv(
    patient_name='adolescent#003',
    reward_fun=reward_fun,
    seed=42,
    max_episode_days=1,
    allow_early_termination=False
)

# Run one episode
obs, _ = env.reset()
done = False
step = 0

glucose_history = []
insulin_history = []
reward_history = []

print("\nRunning one episode (first 50 steps shown):")
print(f"{'Step':<6} {'Glucose':<10} {'Basal':<10} {'Bolus':<10} {'Reward':<10}")
print("-"*70)

while not done and step < 288:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    bg = info['glucose']
    basal = info['basal']
    bolus = info['bolus']
    
    glucose_history.append(bg)
    insulin_history.append(basal + bolus)
    reward_history.append(reward)
    
    if step < 50:  # Print first 50 steps
        print(f"{step:<6} {bg:<10.1f} {basal:<10.3f} {bolus:<10.3f} {reward:<10.3f}")
    
    step += 1

print("\n" + "="*70)
print("EPISODE SUMMARY")
print("="*70)
print(f"Total Steps: {len(glucose_history)}")
print(f"Total Reward: {sum(reward_history):.2f}")
print(f"Mean Reward/Step: {np.mean(reward_history):.3f}")
print(f"\nGlucose Stats:")
print(f"  Mean: {np.mean(glucose_history):.1f} mg/dL")
print(f"  Min:  {np.min(glucose_history):.1f} mg/dL")
print(f"  Max:  {np.max(glucose_history):.1f} mg/dL")
print(f"  Std:  {np.std(glucose_history):.1f} mg/dL")
print(f"\nInsulin Stats:")
print(f"  Mean Total: {np.mean(insulin_history):.3f} U/step")
print(f"  Min:  {np.min(insulin_history):.3f} U/step")
print(f"  Max:  {np.max(insulin_history):.3f} U/step")
print(f"\nTime in Range:")
print(f"  70-180 mg/dL: {((np.array(glucose_history) >= 70) & (np.array(glucose_history) <= 180)).sum() / len(glucose_history) * 100:.1f}%")
print(f"  <70 mg/dL (Hypo): {(np.array(glucose_history) < 70).sum() / len(glucose_history) * 100:.1f}%")
print(f"  >180 mg/dL (Hyper): {(np.array(glucose_history) > 180).sum() / len(glucose_history) * 100:.1f}%")

env.close()
