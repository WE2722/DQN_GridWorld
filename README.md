# DQN GridWorld Visualizer

[![CI](https://github.com/WE2722/DQN_GridWorld/actions/workflows/ci.yml/badge.svg)](https://github.com/WE2722/DQN_GridWorld/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Professional Streamlit app for experimenting with DQN variants on a configurable GridWorld.
This project is designed for interactive research, teaching, and rapid prototyping of
reinforcement-learning agents and simple adversarial defenders.

**Repository:** https://github.com/WE2722/DQN_GridWorld  
**Author:** WE2722

## 🚀 Try it in Your Browser!

Run this app without installing anything:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/WE2722/DQN_GridWorld)
[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/WE2722/DQN_GridWorld)

**No setup required!** GitHub Codespaces provides a complete development environment in your browser with all dependencies pre-installed. See [RUN_IN_BROWSER.md](RUN_IN_BROWSER.md) for detailed instructions.

---

## Table of Contents
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Main Features](#main-features)
- [Testing](#testing)
- [UI Guide](#ui-guide)
- [Example Workflows](#example-workflows)
- [File Overview](#file-overview)
- [Reproducibility & Tips](#reproducibility--tips)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Requirements

- Python 3.9+ (tested on 3.9, 3.10, 3.11)
- Required packages (see `requirements.txt`):
  - streamlit
  - numpy
  - torch
  - pandas
  - matplotlib
  - pillow
  - gym

**Development Dependencies:**
- pytest>=7.0.0 (for running tests)

---

## Quick Start

### Installation (Windows)

**Option 1: Double-click launcher (easiest)**
1. Double-click `Launch_DQN_GridWorld.bat`
2. The app will start automatically and open in your browser

**Option 2: Command line**
```cmd
run_app.bat
```

**Option 3: Custom port**
```cmd
run_app.bat --port 8502
```

### Installation (Manual Setup)

1. Clone the repository:
```bash
git clone https://github.com/WE2722/DQN_GridWorld.git
cd DQN_GridWorld
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

---

## Testing

This project includes a comprehensive test suite with 16 tests covering all core components.

### Running Tests Locally

```bash
# Install test dependencies
pip install pytest>=7.0.0

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_dqn.py
```

### Test Coverage

- **DQN Components** (`tests/test_dqn.py`): 8 tests
  - QNetwork architecture and forward pass
  - ReplayBuffer operations (add, sample)
  - DQNAgent variants (vanilla, deepmind, double)
  
- **GridWorld Environment** (`tests/test_grid_env.py`): 5 tests
  - Environment creation with various configurations
  - Reset functionality
  - Step mechanics and transitions
  - Action space validation
  - Boundary handling

- **Training Functions** (`tests/test_train.py`): 3 tests
  - Training generator functionality
  - Episode completion
  - Reward tracking

### Continuous Integration

All tests run automatically on every push via GitHub Actions:
- ✅ Python 3.9, 3.10, 3.11 compatibility
- ✅ Docker build validation
- ✅ Cross-platform testing (Ubuntu)

View CI status: [GitHub Actions](https://github.com/WE2722/DQN_GridWorld/actions)

---

## Usage Guide

### 1. Environment Configuration
- **Grid Size**: 5x5 to 20x20
- **Goals**: Multiple goals with different reward values
- **Obstacles**: Static barriers with negative rewards
- **Defenders**: Mobile agents that chase the player (tabular or DQN-based)
- **Reset Modes**: Fixed or random goal/obstacle positions

### 2. DQN Variants
- **Vanilla DQN**: Classic Deep Q-Network with experience replay
- **DeepMind DQN**: Enhanced version with target network and optimized hyperparameters
- **Double DQN**: Reduces overestimation using separate action selection and evaluation

### 3. Training & Visualization
- **Live Training**: Real-time episode rewards, Q-value plots, epsilon decay
- **Logs Export**: Download training logs, final Q-table, policy arrays
- **Grid Visualization**: Interactive grid showing agent, goals, obstacles, defenders
- **Episode Replay**: Watch trained agents navigate the environment

---

## Main Features

### 1. Environment Configuration
- **Grid Size**: 5x5 to 20x20
- **Goals**: Multiple goals with different reward values
- **Obstacles**: Static barriers with negative rewards
- **Defenders**: Mobile agents that chase the player (tabular or DQN-based)
- **Reset Modes**: Fixed or random goal/obstacle positions

### 2. DQN Variants
- **Vanilla DQN**: Classic Deep Q-Network with experience replay
- **DeepMind DQN**: Enhanced version with target network and optimized hyperparameters
- **Double DQN**: Reduces overestimation using separate action selection and evaluation

### 3. Training & Visualization
- **Live Training**: Real-time episode rewards, Q-value plots, epsilon decay
- **Logs Export**: Download training logs, final Q-table, policy arrays
- **Grid Visualization**: Interactive grid showing agent, goals, obstacles, defenders
- **Episode Replay**: Watch trained agents navigate the environment

---

## Usage Guide

### Sidebar Parameters

#### Environment
- **Grid size (N x N)**: Larger grids increase the state/action space; learning becomes slower, often needs more episodes or larger networks.
- **Layout seed**: Set ≥ 0 to reproduce the same layout. -1 = random. Useful for deterministic experiments.
- **Max steps / episode**: Episode timeout. Larger values allow longer attempts but may slow training.

#### Entities & Movement
- **Number of goals**: More goals = easier task (higher chance of reaching goal randomly).
- **Number of obstacles**: Obstacles constrain movement. Too many can block feasible paths.
- **Number of defenders**: Adversarial agents that actively attempt to catch the player.
- **Goals move (random)**: Makes the task non-stationary and harder.
- **Obstacles move (random)**: Dynamic obstacles change the navigable map over time.

#### Rewards
- **Reward on reaching goal**: Terminal positive reward; increasing this emphasizes goal-reaching.
- **Reward on defender catch**: Penalty for being caught by a defender.
- **Reward on obstacle collision**: Penalty for colliding with obstacles.
- **Step penalty**: Per-step reward (often negative) to encourage shorter solutions.

#### Training Parameters
- **Episodes**: Number of training episodes per agent.
- **Batch size (DQN)**: Affects gradient variance and compute per update.
- **Replay buffer size**: Larger buffers hold more diverse experiences but require more memory.
- **Learning rate**: Step-size of the optimizer. Too high → unstable; too low → slow.
- **Gamma (discount)**: Future reward discount; near-1 favors long-term returns.
- **Epsilon start/end/decay**: Epsilon-greedy exploration schedule.
- **Target update freq**: For DeepMind/Double DQN, frequency of copying weights to target network.

---

## Features Detail

### Interactive Configuration
- Grid size, number of goals, obstacles, defenders
- Toggle movement for goals and obstacles (random walk)
- Editable reward parameters: goal reward, defender penalty, obstacle penalty, step penalty

### Multiple Agent Training
- Train 1–3 agents in parallel (each on an independent environment copy)
- Choose algorithm per agent: vanilla / deepmind / double
- Side-by-side comparison

### Defender Training
- Per-defender algorithm and hyperparameters (q_learning, sarsa, monte_carlo, td0, dqn)
- Trained defenders act during agent training and evaluation

### Live Visualization
- Grid render per agent during training
- Episode reward and running-mean charts
- Combined overlay chart for direct comparison between agents and defender curves
- Defender training curves displayed in a dedicated panel

### Export & Evaluation
- Download per-episode rewards CSV / JSON for agents and defenders
- Post-training evaluation (greedy agent policy) reports:
  - Average reward
  - Win rate (reaching goal)
  - Catch rate (caught by defender)

---

## Project Structure

```
DQN_GridWorld/
├── app.py                          # Main Streamlit app (UI, config, training, visualization)
├── grid_env.py                     # GridWorld environment with goals, obstacles, defenders
├── dqn.py                          # DQN implementation (QNetwork, ReplayBuffer, DQNAgent)
├── train.py                        # Generator-based training loops
├── defenders.py                    # Defender policy training (tabular + DQN)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker containerization
├── Launch_DQN_GridWorld.bat       # Windows launcher
├── run_app.bat                    # Windows app runner
├── run_app.ps1                    # PowerShell app runner
├── tests/                         # Test suite (pytest)
│   ├── test_dqn.py               # DQN component tests
│   ├── test_grid_env.py          # Environment tests
│   └── test_train.py             # Training function tests
├── .github/workflows/             # CI/CD configuration
│   └── ci.yml                    # GitHub Actions workflow
├── .devcontainer/                 # GitHub Codespaces config
│   └── devcontainer.json
├── .gitpod.yml                    # Gitpod workspace config
├── RUN_IN_BROWSER.md             # Browser-based testing guide
└── README.md                       # This file
```

---

## Algorithm Details

### Agent Algorithms
- **Vanilla DQN**: Single Q-network, targets computed from the same network
- **DeepMind DQN**: Uses a separate target network updated periodically
- **Double DQN**: Policy network selects actions, target network evaluates them (reduces overestimation)

### Defender Training
- Each defender may be trained with its own algorithm and hyperparameters
- Available algorithms: Q-Learning, SARSA, Monte Carlo, TD(0), DQN
- After training, defenders act to catch the agent during training & evaluation

---

## Example Workflows

### 1. Quick Comparison (No Defenders)
- Set grid size to 6, obstacles 5, defenders 0
- Choose two agent slots (vanilla vs deepmind), set episodes to 200
- Start training and watch the live charts and grid renders

### 2. Robustness Test vs Moving Goals
- Enable "Goals move (random)", increase episodes to 500
- Train DeepMind / Double DQN to see which adapts better in non-stationary settings

### 3. Defensive Adversary
- Add 1 defender, set defender algorithm = q_learning and train for 500 episodes
- Enable "Overlay defender curves" to see defender training series with agent curves
- Train agent(s) — trained defender will act to catch during agent training/evaluation

---

## Reproducibility & Best Practices

- Use a non-negative layout seed to reproduce placement of goals/obstacles/defenders
- For consistent comparisons, set the same seed and identical hyperparameters except algorithm choice
- GPU: DQN supports torch and will use CUDA if available. For small grids CPU is usually sufficient

---

## Limitations & Known Issues

- **Observation encoding**: Agents see only normalized (x,y) position — they do not observe defender/goal/obstacle positions explicitly. This is a deliberate simple baseline; extend the observation to include entity positions for better performance.
- **Training runs in Streamlit process**: Long runs may block the UI. Use smaller episode counts for exploratory runs.
- **Defender training**: Currently trains defenders against a random prey baseline (simplified). For full adversarial co-training, implement a joint training loop.

---

## Extending the Project

- **Add richer observations**: Concatenate goal/defender/obstacle positions into the state vector, or provide a grid-shaped image observation for convolutional policies
- **Add prioritized replay**: Implement prioritized experience replay configurable from the UI
- **Model save/load**: Add buttons for saving and loading trained models
- **Multi-agent simultaneous training**: Agents and defenders co-learning

---

## Contributing

Contributions are welcome! Here's how to contribute:

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DQN_GridWorld.git
   cd DQN_GridWorld
   ```

3. Install dependencies (including dev dependencies):
   ```bash
   pip install -r requirements.txt
   pip install pytest>=7.0.0
   ```

4. Run tests to ensure everything works:
   ```bash
   pytest -v
   ```

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests if applicable

3. Run the test suite:
   ```bash
   pytest
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request on GitHub

### CI/CD Pipeline

All pull requests automatically run through our CI pipeline:
- ✅ Tests on Python 3.9, 3.10, 3.11
- ✅ Docker build validation
- ✅ Code quality checks

Your PR must pass all checks before it can be merged.

---

## License

MIT License — see LICENSE file for details.

---

## Contact

- **Repository**: https://github.com/WE2722/DQN_GridWorld
- **Author**: WE2722
