# glucoRL

**Deep Reinforcement Learning for Type 1 Diabetes Blood Glucose Control**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stable Baselines3](https://img.shields.io/badge/RL-Stable%20Baselines3-green.svg)](https://stable-baselines3.readthedocs.io/)

## 📋 Overview

This project implements a **deep reinforcement learning system** for automated insulin delivery in Type 1 Diabetic (T1D) patients. Using the Proximal Policy Optimization (PPO) algorithm, an intelligent agent learns to control an insulin pump to maintain blood glucose levels within the healthy range (70–180 mg/dL).

### The Challenge

Type 1 Diabetes requires constant management of blood glucose levels through insulin administration. Too little insulin leads to **hyperglycaemia** (high blood sugar), causing long-term complications. Too much insulin causes **hypoglycaemia** (low blood sugar), which can be immediately life-threatening. Traditional control methods like fixed basal-bolus regimens or PID controllers struggle to adapt to the complex, non-linear dynamics of glucose metabolism and the variability between patients.

### Our Approach

We train a reinforcement learning agent that:
- **Learns from experience** using FDA-approved virtual patient models (SimGlucose)
- **Anticipates meals** by observing upcoming carbohydrate intake
- **Adapts dynamically** to changing conditions throughout the day
- **Prioritises safety** through an asymmetric reward function that penalises hypoglycaemia more heavily than hyperglycaemia

The agent is evaluated against traditional controllers (Basal-Bolus therapy and PID control) and stress-tested under challenging scenarios like high-carb meals, skipped meals, and irregular eating patterns.

---

## 🏗️ Project Structure

```
RL-Project/
├── src/                          # Source code
│   ├── train_ppo.py              # PPO training script
│   ├── main.py                   # Controller comparison (RL vs BB vs PID)
│   ├── test_robustness.py        # Stress testing under edge cases
│   ├── custom_env.py             # Gymnasium environment for SimGlucose
│   ├── reward_functions.py       # Risk-based reward formulations
│   ├── RL_simulation.py          # Simulation runner for trained agents
│   └── helpers.py                # Utility functions
├── simglucose/                   # SimGlucose simulator (Git submodule)
├── train/                        # Training outputs
│   ├── checkpoints/              # Model checkpoints & best model
│   ├── ppo_tensorboard/          # TensorBoard logs
│   └── results/                  # Final trained models
├── test/                         # Test outputs and results
│   ├── results/                  # Comparison test results
│   ├── robustness/               # Robustness test outputs
│   └── simulations/              # Individual simulation logs
├── environment.yml               # Conda environment specification
└── README.md
```

---

## 🤖 The "Smart" Agent

The RL agent uses a carefully designed observation space that provides rich context for decision-making:

| Observation Component | Description |
|----------------------|-------------|
| **Glucose History** | Last 60 minutes of CGM readings (12 samples at 5-min intervals) |
| **Insulin History** | Last 60 minutes of insulin delivery rates |
| **Meal Lookahead** | Upcoming carbohydrates in the next 2 hours |

### Action Space

The agent outputs a **continuous insulin rate** in the range [-1, 1], which is mapped exponentially to realistic pump values (0–3 U/hr). This exponential mapping allows fine-grained control at low doses while still permitting aggressive dosing when needed.

### Reward Function

Based on the Blood Glucose Risk Index (Kovatchev et al.), our reward function:
- Assigns **zero penalty** when glucose is in range (70–180 mg/dL)
- Applies **asymmetric penalties** — hypoglycaemia is penalised more severely than hyperglycaemia
- Includes a **maximum penalty** for severe hypoglycaemia (<39 mg/dL)

---

## ⚙️ Installation

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git

### Setup

```bash
# Clone the repository with submodules
git clone --recurse-submodules <repo-url>
cd RL-Project

# Create and activate the conda environment
conda env create -f environment.yml
conda activate rl-proj

# Install SimGlucose in development mode
cd simglucose && pip install -e . && cd ..
```

---

## 🚀 Usage

### Training a New Agent

```bash
python src/train_ppo.py
```

Training parameters can be adjusted in the script:
- `n_envs`: Number of parallel environments (default: 4)
- `total_steps`: Total training timesteps (default: 2,000,000)
- `episode_days`: Episode length in days (default: 1)

### Monitoring Training

```bash
tensorboard --logdir train/ppo_tensorboard
```

### Evaluating the Agent

Compare the trained RL agent against Basal-Bolus and PID controllers:

```bash
python src/main.py
```

### Robustness Testing

Test the agent under stress scenarios:

```bash
python src/test_robustness.py
```

This runs four challenging scenarios:
| Scenario | Description |
|----------|-------------|
| **High Carbs** | Large meals (80–110g carbs) to test hyperglycaemia prevention |
| **Missed Lunch** | Skipped meal to test hypoglycaemia avoidance |
| **Late Dinner** | 10:30 PM meal to test nocturnal control |
| **Random Chaos** | Unpredictable meal times and amounts |

---

## 📊 Results

The trained agent is evaluated using standard clinical metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| **TIR** | Time in Range (70–180 mg/dL) | > 70% |
| **TBR** | Time Below Range (<70 mg/dL) | < 4% |
| **TAR** | Time Above Range (>180 mg/dL) | < 25% |
| **CV** | Coefficient of Variation | < 36% |

Results are saved to `test/results/` with detailed CSV data and visualisation plots.

---

## 🔬 SimGlucose Simulator

This project uses [SimGlucose](https://github.com/jxx123/simglucose), an open-source Type 1 Diabetes simulator that implements the FDA-accepted UVA/Padova T1D model. It provides:

- **30 virtual patients**: 10 adults, 10 adolescents, 10 children
- **Realistic glucose dynamics**: Meal absorption, insulin pharmacokinetics
- **CGM sensor models**: With realistic noise and delay
- **Insulin pump models**: With configurable delivery constraints

SimGlucose is included as a Git submodule under `simglucose/`.

---

## 📚 References

- Kovatchev, B. P., et al. "Symmetrization of the Blood Glucose Measurement Scale and Its Applications." *Diabetes Care*, 1997.
- Xie, J. "Simglucose v0.2.1." GitHub, 2018.
- Schulman, J., et al. "Proximal Policy Optimization Algorithms." *arXiv preprint*, 2017.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The SimGlucose simulator (`simglucose/`) is also MIT licensed. See [simglucose/LICENSE](simglucose/LICENSE).

---

## 🙏 Acknowledgements

- [SimGlucose](https://github.com/jxx123/simglucose) by Jinyu Xie for the T1D simulation environment
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for the PPO implementation
- University of Padova and University of Virginia for the original T1D metabolic model

