```markdown
# DQN GridWorld Visualizer

Professional Streamlit app for experimenting with DQN variants on a configurable GridWorld.
This project is designed for interactive research, teaching, and rapid prototyping of
reinforcement-learning agents and simple adversarial defenders.

Repository owner: welhafid-art

Contents
- A configurable GridWorld environment (goals, obstacles, defenders — static or moving).
- DQN implementations:
  - Vanilla DQN (no target network)
  - DeepMind-style DQN (with target network)
  - Double DQN (action selection by policy_net, evaluation by target_net)
- Defender training utilities (tabular methods + DQN) with per-defender algorithm/hyperparameters.
- Streamlit UI to configure hyperparameters, run training, visualize environments and live metrics,
  compare multiple agents, export/import episode rewards, and run short evaluations.

This README documents installation, usage, design, hyperparameters, file structure, limitations,
and recommended next steps so the project is ready to push to your GitHub repository.

Table of contents
- Requirements
- Quick start
- Main features
- UI guide (sidebar explanation & how each hyperparameter influences behavior)
- Example workflows
- File overview
- Reproducibility & tips
- Limitations and recommended extensions
- License

Requirements
- Python 3.8+
- Recommended packages (see requirements.txt):
  - streamlit
  - gym
  - torch
  - numpy
  - pandas
  - Pillow

Install
1. Create and activate a virtual environment (recommended):
   - python -m venv .venv
   - source .venv/bin/activate  (macOS / Linux)
   - .venv\Scripts\activate     (Windows)

2. Install dependencies:
   pip install -r requirements.txt

Quick start (run the app)
1. From the repository root:
   streamlit run app.py

2. The Streamlit UI will open in your browser. Use the sidebar to configure the environment,
   pick algorithms/hyperparameters, and start training.

Main features
- Interactive configuration:
  - Grid size, number of goals, obstacles, defenders.
  - Toggle movement for goals and obstacles (random walk).
  - Editable reward parameters: goal reward, defender penalty, obstacle penalty, step penalty.
- Multiple agent training:
  - Train 1–3 agents in parallel (each on an independent environment copy).
  - Choose algorithm per agent: vanilla / deepmind / double.
- Defender training:
  - Per-defender algorithm and hyperparameters (q_learning, sarsa, monte_carlo, td0, dqn).
  - Trained defenders are attached to agent envs and act (attempt to catch agent) during agent training and evaluation.
- Live visualization:
  - Grid render per agent during training.
  - Episode reward and running-mean charts.
  - Combined overlay chart for direct comparison between agents and defender curves.
  - Defender training curves displayed in a dedicated panel.
- Export / edit rewards:
  - Download per-episode rewards CSV / JSON for agents and defenders.
  - In-app editable table (if Streamlit version supports experimental_data_editor) to tweak reward series.
  - (Uploader removed by request — re-import is intentionally outside the main UI; consult the app notes).
- Evaluation / scoring:
  - Post-training evaluation (greedy agent policy) reports avg_reward, win_rate (reaching goal), catch_rate (caught by defender).

Sidebar parameter guide — what each option does and how it influences learning
Environment
- Grid size (N x N)
  - Larger grids increase the state/action space: learning becomes slower, often needs more episodes or larger networks.
- Layout seed (-1 = random)
  - Set ≥ 0 to reproduce the same layout. Useful for deterministic experiments and comparisons.
- Max steps / episode
  - Episode timeout. A larger value allows longer attempts but may slow training and reduce signal for short-path tasks.

Entities & movement
- Number of goals
  - More goals = more success states -> easier task (higher chance of reaching goal randomly).
- Number of obstacles
  - Obstacles constrain movement. Too many can block feasible paths.
- Number of defender agents
  - Adversarial agents; when trained and attached they actively attempt to catch the agent.
- Goals move (random)
  - When enabled, goals change position randomly each step — makes the task non-stationary and harder.
- Obstacles move (random)
  - Dynamic obstacles change the navigable map over time.

Rewards (editable defaults)
- Reward on reaching goal
  - Terminal positive reward; increasing this emphasizes goal-reaching.
- Reward on defender catch (negative)
  - Penalty for being caught by a defender; larger negative values strongly discourage being caught.
- Reward on obstacle collision (negative)
  - Penalty for colliding with obstacles.
- Step penalty
  - Per-step reward (often negative) to encourage shorter solutions (faster goal attainment).

Training (agents)
- Episodes
  - Number of training episodes per agent.
- Render every N episodes
  - How frequently to render frames for the UI. Higher values reduce UI overhead.
- Batch size (DQN)
  - DQN mini-batch size. Affects gradient variance and compute per update.
- Replay buffer size (DQN)
  - Larger buffers hold more diverse experiences (useful) but require more memory.
- Learning rate (DQN)
  - Step-size of the optimizer. Too high → unstable; too low → slow.
- Gamma (discount)
  - Future reward discount; near-1 favors long-term returns.
- Epsilon start / end / decay
  - Epsilon-greedy exploration schedule. Slower decay means more exploration for longer.
- Target update freq
  - For algorithms that use a target network (deepmind/double), frequency of copying weights to target.

Agents to train
- Number of agent slots (1–3)
  - Each slot is an independent agent trained in its own environment copy, allowing side-by-side comparisons.
- Agent algorithm selection (per slot)
  - "vanilla": Single Q-network, targets computed from the same network.
  - "deepmind": Uses a separate target network updated periodically.
  - "double": Double DQN — policy_net selects next actions, target_net evaluates them (reduces overestimation).

Defenders
- Per-defender selection
  - Each defender may be trained with its own algorithm and hyperparameters.
  - After training, defenders are attached (via policy wrappers) to agent environment copies and act to catch the agent during agent training & evaluation.

Example workflows
1. Quick comparison (no defenders)
   - Set grid size to 6, obstacles 5, defenders 0.
   - Choose two agent slots (vanilla vs deepmind), set episodes to 200.
   - Start training and watch the live charts and grid renders.

2. Robustness test vs moving goals
   - Enable "Goals move (random)", increase episodes to 500.
   - Train DeepMind / Double DQN to see which adapts better in a non-stationary setting.

3. Defensive adversary
   - Add 1 defender, set defender algorithm = q_learning and train for 500 episodes.
   - Enable "Overlay defender curves" if you want to see defender training series together with agent curves.
   - Train agent(s) — trained defender will act to catch during agent training/evaluation.

File overview
- app.py — main Streamlit app: orchestrates UI, config, training, visualization, export.
- grid_env.py — GridWorld environment (gym.Env) with support for multiple goals, obstacles, defenders, movement flags and configurable rewards.
- dqn.py — DQN implementation: QNetwork, ReplayBuffer, DQNAgent supporting "vanilla", "deepmind", "double".
- train.py — generator-based training loops: train_generator and compare_train_generator (yields progress so Streamlit updates).
- defenders.py — helpers to train defender policies (tabular algorithms + DQN) and policy wrappers for runtime use.
- requirements.txt — Python package requirements for the app.

Reproducibility & best practices
- Use a non-negative layout seed to reproduce placement of goals/obstacles/defenders.
- For consistent comparisons between algorithms set the same seed and identical hyperparameters except the algorithm choice.
- GPU: DQN supports torch and will use CUDA device if available. For small grids CPU is usually sufficient.

Limitations and known issues
- Observation encoding: currently agents see only the normalized (x,y) position of the agent — they do not observe defender/goal/obstacle positions explicitly. This is a deliberate simple baseline; extend the observation to include entity positions for better performance.
- Training runs in the Streamlit process: long runs may block the UI. Use smaller episode counts for exploratory runs, or adapt the code to run training in a background process.
- Defender training currently trains defenders against a random prey baseline (simplified). For full adversarial co-training, implement a joint training loop (advanced feature).
- Upload/import of edited rewards is intentionally removed from the UI (per request). You may re-import rewards using developer workflows if required.

Extending the project
- Add richer observations:
  - Concatenate goal/defender/obstacle positions into the state vector, or provide a grid-shaped image observation for convolutional policies.
- Add prioritized replay and Double DQN variants configurable from the UI.
- Add model save/load buttons and repeatable evaluation scripts.
- Implement multi-agent simultaneous training (agents and defenders co-learning).


License
- MIT License — see LICENSE file (if you want I can add an explicit license file).

Contact
- Repository owner: welhafid-art
- For changes you'd like added (UI tweaks, different defender training logic, adding observation features), open an issue or ask in a message and I will prepare a patch or PR.
