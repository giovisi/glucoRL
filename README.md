# RL for Type 1 Diabetes Blood Glucose Control

Deep reinforcement learning for automated insulin delivery in T1D patients. A PPO agent learns to control an insulin pump using the SimGlucose simulator (FDA-approved virtual patient models).

## What's in here

The `src/` folder contains all the code:
- `train_ppo.py` – trains the PPO agent
- `main.py` – compares RL vs Basal-Bolus vs PID controllers
- `test_robustness.py` – stress tests (high carbs, skipped meals, etc.)
- `custom_env.py` – Gymnasium environment wrapping SimGlucose
- `reward_functions.py` – risk-based reward functions

SimGlucose is included as a Git submodule.

## Setup

```bash
git clone --recurse-submodules <repo-url>
conda env create -f environment.yml
conda activate rl-proj
cd simglucose && pip install -e . && cd ..
```

## Running

```bash
# Train
python src/train_ppo.py

# Evaluate
python src/main.py

# Robustness tests
python src/test_robustness.py

# Monitor training
tensorboard --logdir train/ppo_tensorboard
```

## The agent

The "Smart" agent observes:
- Last hour of glucose readings
- Last hour of insulin delivery
- Upcoming carbs in the next 2 hours (meal lookahead)

Actions are continuous insulin rates, mapped exponentially to realistic values (0–3 U/hr).

