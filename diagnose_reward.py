import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMODULE_ROOT = os.path.join(PROJECT_ROOT, "simglucose")
sys.path.insert(0, SUBMODULE_ROOT)

from custom_env import CustomT1DEnv, paper_reward, compute_risk_index

def test_reward_function():
    """Test the reward function across different glucose values"""
    print("="*70)
    print("REWARD FUNCTION ANALYSIS")
    print("="*70)
    
    # Test glucose range
    bg_values = [30, 39, 40, 50, 70, 100, 140, 180, 250, 350, 400]
    
    print("\nReward values for different glucose levels:")
    print(f"{'BG (mg/dL)':<15} {'Raw RI':<15} {'Normalized RI':<20} {'Reward':<15}")
    print("-"*70)
    
    for bg in bg_values:
        raw_ri = compute_risk_index(bg)
        reward = paper_reward([bg])
        normalized_ri = min(raw_ri / 100.0, 1.0)
        
        print(f"{bg:<15.1f} {raw_ri:<15.2f} {normalized_ri:<20.4f} {reward:<15.4f}")
    
    # Visualize reward landscape
    bg_range = np.linspace(20, 400, 200)
    rewards = [paper_reward([bg]) for bg in bg_range]
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Reward function
    plt.subplot(1, 2, 1)
    plt.plot(bg_range, rewards, linewidth=2, color='blue')
    plt.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Zero Reward')
    plt.axvline(x=70, color='red', linestyle='--', alpha=0.5, label='Hypo Threshold')
    plt.axvline(x=180, color='orange', linestyle='--', alpha=0.5, label='Hyper Threshold')
    plt.axhline(y=-15, color='darkred', linestyle=':', alpha=0.5, label='Max Penalty')
    plt.fill_between([70, 180], -20, 5, alpha=0.2, color='green', label='Target Range')
    plt.xlabel('Blood Glucose (mg/dL)')
    plt.ylabel('Reward')
    plt.title('Reward Function Landscape')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([-16, 1])
    
    # Plot 2: Risk Index
    plt.subplot(1, 2, 2)
    risk_values = [compute_risk_index(bg) for bg in bg_range]
    plt.plot(bg_range, risk_values, linewidth=2, color='red')
    plt.axvline(x=70, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=180, color='orange', linestyle='--', alpha=0.5)
    plt.fill_between([70, 180], 0, max(risk_values), alpha=0.2, color='green')
    plt.xlabel('Blood Glucose (mg/dL)')
    plt.ylabel('Risk Index')
    plt.title('Raw Risk Index')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reward_analysis.png', dpi=150)
    print("\n✓ Saved reward visualization to: reward_analysis.png")
    plt.close()

def test_random_agent():
    """Test a random agent to see typical episode lengths and rewards"""
    print("\n" + "="*70)
    print("RANDOM AGENT BASELINE TEST")
    print("="*70)
    
    env = CustomT1DEnv(patient_name='adolescent#003', seed=42)
    
    n_episodes = 5
    episode_lengths = []
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        glucose_history = []
        
        while not done and ep_length < 288:  # Max 1 day
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            glucose_history.append(info.get('glucose', 0))
            
            if done:
                break
        
        episode_lengths.append(ep_length)
        episode_rewards.append(ep_reward)
        
        print(f"\nEpisode {ep+1}:")
        print(f"  Length: {ep_length} steps")
        print(f"  Total Reward: {ep_reward:.2f}")
        print(f"  Mean Reward/Step: {ep_reward/ep_length:.4f}")
        if glucose_history:
            print(f"  Final Glucose: {glucose_history[-1]:.1f} mg/dL")
            print(f"  Mean Glucose: {np.mean(glucose_history):.1f} mg/dL")
            print(f"  Min Glucose: {np.min(glucose_history):.1f} mg/dL")
            print(f"  Max Glucose: {np.max(glucose_history):.1f} mg/dL")
    
    print("\n" + "-"*70)
    print("RANDOM AGENT SUMMARY:")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Mean Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    env.close()

def analyze_action_space():
    """Analyze the action space to understand insulin dosing"""
    print("\n" + "="*70)
    print("ACTION SPACE ANALYSIS")
    print("="*70)
    
    env = CustomT1DEnv(patient_name='adolescent#003', seed=42)
    
    print(f"\nAction Space: {env.action_space}")
    print(f"  Low:  {env.action_space.low}")
    print(f"  High: {env.action_space.high}")
    print(f"  Shape: {env.action_space.shape}")
    
    print(f"\nInterpretation:")
    print(f"  Action[0] = Basal Rate (0 to {env.action_space.high[0]:.2f} U/min)")
    print(f"  Action[1] = Bolus Dose (0 to {env.action_space.high[1]:.2f} U)")
    
    # Test a few actions
    print("\nTesting sample actions:")
    obs, _ = env.reset()
    
    test_actions = [
        [0.0, 0.0],  # No insulin
        [0.05, 0.0],  # Small basal
        [0.1, 0.0],  # Medium basal
        [0.0, 2.0],  # Medium bolus
        [0.1, 3.0],  # Both
    ]
    
    for i, action in enumerate(test_actions):
        obs, _ = env.reset()
        obs, reward, done, truncated, info = env.step(np.array(action, dtype=np.float32))
        print(f"  Action {action}: Reward = {reward:.4f}, BG = {info.get('glucose', 0):.1f} mg/dL")
    
    env.close()

def suggest_improvements():
    """Suggest hyperparameter and reward modifications"""
    print("\n" + "="*70)
    print("SUGGESTED IMPROVEMENTS")
    print("="*70)
    
    print("\n1. REWARD FUNCTION ISSUES:")
    print("   - The -15 penalty for BG <= 39 might be too harsh initially")
    print("   - Consider a softer penalty schedule or curriculum learning")
    print("   - The normalized RI might have poor gradient properties")
    
    print("\n2. HYPERPARAMETER SUGGESTIONS:")
    print("   a) Increase exploration (ent_coef):")
    print("      Current: 0.005 → Try: 0.01 or 0.02")
    print("      Reason: Agent needs to explore more actions")
    
    print("\n   b) Adjust learning rate:")
    print("      Current: 3e-4 → Try: 1e-4 (slower) or 5e-4 (faster)")
    print("      Reason: Fine-tune convergence speed")
    
    print("\n   c) Increase batch size:")
    print("      Current: 256 → Try: 512 or 1024")
    print("      Reason: More stable gradient updates")
    
    print("\n   d) Reduce gamma (discount factor):")
    print("      Current: 0.995 → Try: 0.99 or 0.98")
    print("      Reason: Focus more on immediate rewards")
    
    print("\n   e) More training steps per update:")
    print("      Current: n_steps=2048 → Try: 4096")
    print("      Reason: Collect more experience before updating")
    
    print("\n3. ALTERNATIVE REWARD FUNCTIONS TO TEST:")
    print("   a) Shaped reward with intermediate goals")
    print("   b) Curriculum learning (start with easier scenarios)")
    print("   c) Reward shaping with potential-based rewards")

if __name__ == "__main__":
    print("Starting diagnostic analysis...\n")
    
    test_reward_function()
    test_random_agent()
    analyze_action_space()
    suggest_improvements()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
