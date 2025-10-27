# app.py
import streamlit as st
import numpy as np
import torch
import random
import copy
import io
import csv
import json
import pandas as pd
from grid_env import GridWorldEnv
from dqn import DQNAgent
from train import compare_train_generator
from defenders import train_defender_tabular, train_defender_dqn, _find_free_cell, TabularPolicyWrapper, DQNPolicyWrapper

st.set_page_config(layout="wide", page_title="DQN GridWorld Visualizer (Editable rewards & moving entities)")

st.title("DQN GridWorld Visualizer — editable rewards, moving goals/obstacles, defender policies")

# ------------------------
# Sidebar: environment & core training params + rewards
# All controls now include concise descriptions (help=) explaining how they influence behavior.
# ------------------------
with st.sidebar:
    st.header("Environment")
    size = st.number_input(
        "Grid size (N x N)",
        min_value=4,
        max_value=20,
        value=6,
        help="Grid width/height (NxN). Larger grids increase the state space and usually require more training episodes or stronger function approximators."
    )
    seed = st.number_input(
        "Layout seed (-1 = random)",
        value=-1,
        step=1,
        help="Deterministic seed for placing goals/obstacles/defenders. Set >=0 for reproducible layouts; -1 samples a new random layout."
    )
    max_steps = st.number_input(
        "Max steps / episode",
        min_value=5,
        max_value=1000,
        value=50,
        help="Maximum number of steps before an episode ends (timeout). Larger values allow longer episodes but may slow learning."
    )

    st.markdown("---")
    st.header("Entities & movement")
    num_goals = st.number_input(
        "Number of goals",
        min_value=1,
        max_value=6,
        value=1,
        help="How many goal cells exist. More goals usually make the task easier because there are more successful terminal states."
    )
    num_obstacles = st.number_input(
        "Number of obstacles",
        min_value=0,
        max_value=(size*size - 2),
        value=5,
        help="Number of obstacles placed on the grid. Obstacles block movement and increase path-planning difficulty."
    )
    num_defenders = st.number_input(
        "Number of defender agents",
        min_value=0,
        max_value=6,
        value=0,
        help="Number of adversarial agents. If trained, they will try to catch the main agents during training/evaluation."
    )

    moving_goals = st.checkbox(
        "Goals move (random)",
        value=False,
        help="If enabled, goal positions perform a random walk each step. Makes the task non-stationary and harder."
    )
    moving_obstacles = st.checkbox(
        "Obstacles move (random)",
        value=False,
        help="If enabled, obstacles perform a random walk each step. Dynamic obstacles change feasible paths over time."
    )

    st.markdown("---")
    st.header("Rewards (editable defaults)")
    reward_goal = st.number_input(
        "Reward on reaching goal",
        value=1.0,
        format="%.4f",
        help="Reward given when the agent reaches any goal. Increase to emphasize reaching goals."
    )
    reward_defender = st.number_input(
        "Reward on defender catch (negative)",
        value=-1.0,
        format="%.4f",
        help="Reward (typically negative) when agent collides with a defender. Lower (more negative) penalizes being caught more strongly."
    )
    reward_obstacle = st.number_input(
        "Reward on obstacle collision (negative)",
        value=-1.0,
        format="%.4f",
        help="Reward (negative) applied if the agent collides with an obstacle. Use to discourage running into obstacles."
    )
    reward_step = st.number_input(
        "Step penalty",
        value=-0.01,
        format="%.4f",
        help="Per-step reward (usually negative) to encourage shorter trajectories. Larger negative values force faster solutions."
    )

    st.markdown("---")
    st.header("Training (Agents)")
    episodes = st.number_input(
        "Episodes (per agent)",
        min_value=1,
        max_value=5000,
        value=300,
        help="Number of training episodes for each selected agent. More episodes -> more training time and usually better final performance."
    )
    render_every = st.number_input(
        "Render every N episodes",
        min_value=1,
        max_value=100,
        value=1,
        help="How often to produce the environment frame for visualization. Larger values reduce UI overhead during training."
    )
    batch_size = st.number_input(
        "Batch size (DQN)",
        min_value=1,
        max_value=512,
        value=32,
        help="Mini-batch size sampled from replay buffer for DQN updates. Larger batches reduce gradient variance but cost more compute."
    )
    replay_size = st.number_input(
        "Replay buffer size (DQN)",
        min_value=100,
        max_value=200000,
        value=5000,
        help="Capacity of the experience replay buffer. Larger buffers store more diverse experiences but use more memory."
    )
    lr = st.number_input(
        "Learning rate (DQN)",
        min_value=1e-5,
        max_value=1.0,
        value=1e-3,
        format="%.6f",
        help="Optimizer learning rate. High values can destabilize training; low values make learning slow."
    )
    gamma = st.number_input(
        "Gamma (discount)",
        min_value=0.0,
        max_value=1.0,
        value=0.99,
        format="%.3f",
        help="Discount factor for future rewards. Near 1 gives long-term credit; near 0 makes agent short-sighted."
    )
    epsilon_start = st.number_input(
        "Epsilon start",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        format="%.3f",
        help="Initial epsilon for epsilon-greedy exploration. Higher = more random actions at start."
    )
    epsilon_end = st.number_input(
        "Epsilon end",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        format="%.3f",
        help="Final epsilon after decay. Lower values mean less exploration at the end of training."
    )
    epsilon_decay = st.number_input(
        "Epsilon decay (steps)",
        min_value=1,
        max_value=100000,
        value=500,
        step=1,
        help="Controls how quickly epsilon decays from start->end. Larger values keep exploration longer."
    )
    target_update = st.number_input(
        "Target update freq (steps)",
        min_value=1,
        max_value=10000,
        value=100,
        help="For target-net algorithms: how often to copy policy network weights to the target network. Larger values give a more stable but slower-to-update target."
    )

    st.markdown("---")
    st.header("Agents to train")
    agent_slots = st.selectbox(
        "Number of agent slots",
        options=[1,2,3],
        index=2,
        help="How many independent agents (and env copies) to train & compare in parallel."
    )
    agent_algos = []
    for i in range(agent_slots):
        algo = st.selectbox(
            f"Agent {i+1} algorithm",
            options=["vanilla", "deepmind", "double"],
            index=0,
            key=f"agent_algo_{i}",
            help="Choose algorithm variant: 'vanilla' (no target net), 'deepmind' (target net), or 'double' (Double DQN)."
        )
        agent_algos.append(algo)

    st.markdown("---")
    st.header("Defenders (per-defender algorithms & hyperparams)")
    defender_configs = []
    for i in range(num_defenders):
        st.markdown(f"Defender #{i+1}")
        da = st.selectbox(
            f"Defender {i+1} algorithm",
            options=["none (stationary)", "q_learning", "sarsa", "monte_carlo", "td0", "dqn"],
            index=0,
            key=f"def_algo_{i}",
            help="Choose how this defender will be trained. 'none' keeps it stationary; tabular methods suit small grids; 'dqn' uses function approximation."
        )
        d_eps = st.number_input(
            f"Defender {i+1} training episodes",
            min_value=1,
            max_value=2000,
            value=200,
            key=f"def_eps_{i}",
            help="Number of episodes to train this defender policy."
        )
        d_alpha = st.number_input(
            f"Defender {i+1} alpha (tabular)",
            min_value=1e-4,
            max_value=1.0,
            value=0.2,
            key=f"def_alpha_{i}",
            help="Step-size (alpha) used by tabular algorithms (Q-learning/SARSA/TD)."
        )
        d_lr = st.number_input(
            f"Defender {i+1} lr (DQN)",
            min_value=1e-5,
            max_value=1.0,
            value=1e-3,
            key=f"def_lr_{i}",
            help="Learning rate used when training defender with DQN."
        )
        defender_configs.append({"algo": da, "episodes": int(d_eps), "alpha": float(d_alpha), "lr": float(d_lr)})

    st.markdown("---")
    include_defender_in_comparison = st.checkbox(
        "Overlay defender curves in combined chart (if trained)",
        value=True,
        help="If checked, defender training curves will be overlaid on the combined agents chart during agent training; otherwise shown separately afterwards."
    )
    st.markdown("---")
    start_training = st.button("Start training")

# ------------------------
# Environment builder (uses rewards + movement flags)
# ------------------------
def _build_env():
    sval = int(seed) if seed is not None and seed >= 0 else None
    if sval is not None:
        random.seed(sval)
        np.random.seed(sval)
    cells = [(x, y) for x in range(size) for y in range(size)]
    forbidden = {tuple((0,0))}
    def sample_positions(k):
        avail = [c for c in cells if c not in forbidden]
        random.shuffle(avail)
        picks = []
        for _ in range(k):
            if not avail:
                break
            p = avail.pop()
            picks.append(p)
            forbidden.add(p)
        return picks

    goals = sample_positions(num_goals)
    obstacles = sample_positions(num_obstacles)
    defenders_positions = sample_positions(num_defenders)

    defenders_list = [{'pos': p, 'moving': False, 'policy': None} for p in defenders_positions]
    goals_entities = [{'pos': p, 'moving': moving_goals} for p in goals]
    obstacles_entities = [{'pos': p, 'moving': moving_obstacles} for p in obstacles]

    env_local = GridWorldEnv(
        size=size,
        start=(0,0),
        goals=goals_entities,
        obstacles=obstacles_entities,
        defenders=defenders_list,
        max_steps=max_steps,
        seed=sval,
        reward_goal=reward_goal,
        reward_defender=reward_defender,
        reward_obstacle=reward_obstacle,
        reward_step=reward_step,
    )
    return env_local

current_params = {
    "size": int(size),
    "seed": int(seed),
    "max_steps": int(max_steps),
    "num_goals": int(num_goals),
    "num_obstacles": int(num_obstacles),
    "num_defenders": int(num_defenders),
    "moving_goals": bool(moving_goals),
    "moving_obstacles": bool(moving_obstacles),
    "reward_goal": float(reward_goal),
    "reward_defender": float(reward_defender),
    "reward_obstacle": float(reward_obstacle),
    "reward_step": float(reward_step),
}
if ("env_params" not in st.session_state) or (st.session_state["env_params"] != current_params) or ("env" not in st.session_state):
    st.session_state["env"] = _build_env()
    st.session_state["env_params"] = current_params

env = st.session_state["env"]

# ------------------------
# Main layout
# ------------------------
col_env, col_metrics = st.columns([1,2])
with col_env:
    st.subheader("Environment preview")
    img_holder = st.empty()
    try:
        img_holder.image(env.render(mode="rgb_array"), caption="Grid preview", use_container_width=True)
    except Exception as e:
        st.error(f"Render error: {e}")
    st.write(f"Grid size: {env.size} | Max steps: {env.max_steps}")
    st.write(f"Goals: {len(env.goals)} | Obstacles: {len(env.obstacles)} | Defenders: {len(env.defenders)}")
    st.write(f"Rewards: goal={env.reward_goal}, defender={env.reward_defender}, obstacle={env.reward_obstacle}, step={env.reward_step}")

with col_metrics:
    st.subheader("Live metrics")
    ep_text = st.empty()

# ------------------------
# Training flow
# ------------------------
if start_training:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) Train defenders independently as requested, producing a policy wrapper per defender
    defender_wrappers = [None] * num_defenders
    defender_logs_all = [None] * num_defenders
    for i, cfg in enumerate(defender_configs):
        if cfg["algo"] == "none (stationary)":
            defender_wrappers[i] = None
            defender_logs_all[i] = None
            continue
        st.info(f"Training Defender #{i+1} with {cfg['algo']} for {cfg['episodes']} episodes...")
        # make a training copy (preserve reward params)
        layout = {
            'size': env.size,
            'start': env.start,
            'goals': copy.deepcopy([{'pos': g['pos'], 'moving': g['moving']} for g in env.goals]),
            'obstacles': copy.deepcopy([{'pos': o['pos'], 'moving': o['moving']} for o in env.obstacles]),
            'defenders': copy.deepcopy([{'pos': d['pos'], 'moving': d.get('moving', False), 'policy': None} for d in env.defenders]),
            'max_steps': env.max_steps,
            'seed': env._seed_val
        }
        train_env = GridWorldEnv(size=layout['size'], start=layout['start'], goals=layout['goals'], obstacles=layout['obstacles'], defenders=layout['defenders'], max_steps=layout['max_steps'], seed=layout['seed'],
                                 reward_goal=env.reward_goal, reward_defender=env.reward_defender, reward_obstacle=env.reward_obstacle, reward_step=env.reward_step)
        # add a temp defender if none exist in training copy
        if not getattr(train_env, "defenders", None) or len(train_env.defenders) == 0:
            pos = _find_free_cell(train_env)
            train_env.defenders = [{'pos': pos, 'moving': False, 'policy': None}]
            train_env.reset()
            st.warning(f"No defenders in layout: added a temporary defender at {pos} for Defender #{i+1} training copy.")
        try:
            if cfg["algo"] in ("q_learning", "sarsa", "monte_carlo", "td0"):
                wrapper, logs = train_defender_tabular(env=train_env, defender_index=0, episodes=cfg["episodes"], max_steps=max_steps, algo=cfg["algo"], alpha=cfg["alpha"], gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay, render_every=max(1, int(render_every)))
                defender_wrappers[i] = wrapper
                defender_logs_all[i] = logs
            elif cfg["algo"] == "dqn":
                wrapper, logs = train_defender_dqn(env=train_env, defender_index=0, episodes=cfg["episodes"], max_steps=max_steps, lr=cfg["lr"], gamma=gamma, batch_size=int(batch_size), replay_size=int(replay_size), epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay, target_update=int(target_update), device=device, render_every=max(1, int(render_every)))
                defender_wrappers[i] = wrapper
                defender_logs_all[i] = logs
            st.success(f"Defender #{i+1} training finished.")
            # do not plot defender logs now (avoid duplicate); we'll always display them after training in a dedicated panel
        except Exception as e:
            st.error(f"Defender #{i+1} training failed: {e}")
            st.stop()

    # 2) Prepare agent env copies and attach defender policies (so defenders will act to catch)
    layout = {
        'size': env.size,
        'start': env.start,
        'goals': copy.deepcopy([{'pos': g['pos'], 'moving': g.get('moving', False)} for g in env.goals]),
        'obstacles': copy.deepcopy([{'pos': o['pos'], 'moving': o.get('moving', False)} for o in env.obstacles]),
        'defenders': copy.deepcopy([{'pos': d['pos'], 'moving': False, 'policy': None} for d in env.defenders]),
        'max_steps': env.max_steps,
        'seed': env._seed_val
    }

    # build per-agent envs and agents; pass reward params
    envs = []
    agents = []
    labels = []
    def make_agent(variant_name):
        return DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, lr=float(lr), gamma=float(gamma), batch_size=int(batch_size), replay_size=int(replay_size), device=device, epsilon_start=float(epsilon_start), epsilon_end=float(epsilon_end), epsilon_decay=int(epsilon_decay), variant=variant_name, target_update_freq=int(target_update))

    for i, algo in enumerate(agent_algos):
        env_copy = GridWorldEnv(size=layout['size'], start=layout['start'], goals=copy.deepcopy(layout['goals']), obstacles=copy.deepcopy(layout['obstacles']), defenders=copy.deepcopy(layout['defenders']), max_steps=layout['max_steps'], seed=layout['seed'],
                                reward_goal=env.reward_goal, reward_defender=env.reward_defender, reward_obstacle=env.reward_obstacle, reward_step=env.reward_step)
        # Attach wrappers to declared defenders (env_copy.defenders) BEFORE reset so runtime defenders inherit policies
        for di in range(len(env_copy.defenders)):
            if di < len(defender_wrappers) and defender_wrappers[di] is not None:
                env_copy.defenders[di]['policy'] = defender_wrappers[di]
                env_copy.defenders[di]['moving'] = True
            else:
                env_copy.defenders[di]['policy'] = None
                env_copy.defenders[di]['moving'] = False
        # now reset to initialize runtime copies with policy attached
        env_copy.reset()

        envs.append(env_copy)
        labels.append(f"Agent {i+1} ({algo})")
        try:
            agents.append(make_agent(algo))
        except Exception as e:
            st.error(f"Failed to create agent {i+1} with algo '{algo}': {e}")
            st.stop()

    # 3) Create UI columns and run compare_train_generator
    n = len(agents)
    cols = st.columns([1]*n + [1.2])
    img_holders = []
    reward_charts = []
    avg_charts = []
    loss_texts = []
    for i in range(n):
        with cols[i]:
            st.subheader(labels[i])
            img_holders.append(st.empty())
            reward_charts.append(st.line_chart())
            avg_charts.append(st.line_chart())
            loss_texts.append(st.empty())
    with cols[-1]:
        st.subheader("Comparison and Scores")
        st.markdown(
            """
            **Plot descriptions**
            - Episode reward (per-agent): total reward per training episode (higher is better).
            - Mean reward (running): running average over recent episodes (default 100) to show trend/stability.
            - Combined chart: overlays rewards from all selected agents and trained defenders (if any) for direct comparison.
            - Defender curves: show defender training reward per episode (if defenders were trained).
            - Scores: after training an evaluation is run and shows avg_reward, win_rate (agent reaches goal), catch_rate (defender catches agent).
            """
        )
        combined_chart = st.line_chart()
        combined_avg = st.line_chart()
        score_area = st.empty()
        prog = st.progress(0)

    gen = compare_train_generator(envs=envs, agents=agents, episodes=int(episodes), max_steps_per_episode=int(max_steps), render_every=int(render_every), update_every=1, yield_every_episode=1)

    agent_training_logs = [[] for _ in range(n)]
    try:
        for update in gen:
            ep = update['episode']
            ep_text.markdown(f"Episode: **{ep}** (running)")
            for i, info in enumerate(update['agents']):
                if info['frame'] is not None:
                    img_holders[i].image(info['frame'], caption=f"{labels[i]} Ep {ep}", use_container_width=True)
                loss_texts[i].text(f"Last loss: {info['loss']:.6f}" if info['loss'] is not None else "Last loss: -")
                reward_charts[i].add_rows({"episode_reward": [info['episode_reward']]})
                avg_charts[i].add_rows({"mean_reward": [info['mean_reward']]})
                agent_training_logs[i].append(info['episode_reward'])

            combined_row = {}
            combined_avg_row = {}
            for i, lab in enumerate(labels):
                combined_row[f"{lab} reward"] = update['agents'][i]['episode_reward']
                combined_avg_row[f"{lab} mean"] = update['agents'][i]['mean_reward']

            # optionally overlay defender training curves (per-defender)
            if include_defender_in_comparison:
                for di, logs in enumerate(defender_logs_all):
                    if logs is not None:
                        dval = logs[ep-1] if (ep-1) < len(logs) else logs[-1]
                        combined_row[f"Defender {di+1} reward"] = float(dval)
                        combined_avg_row[f"Defender {di+1} mean"] = float(dval)

            combined_chart.add_rows(combined_row)
            combined_avg.add_rows(combined_avg_row)
            prog.progress(min(ep / int(episodes), 1.0))

    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    st.success("Training finished for selected agents.")

    # store last rewards in session_state for download / reuse
    st.session_state['last_rewards'] = {
        "agents": agent_training_logs,
        "defenders": defender_logs_all
    }

    # 4) Evaluation / scoring: run evaluation_episodes per agent env with greedy agent policy and defenders active
    eval_episodes = min(100, max(10, int(episodes//3)))
    st.subheader("Evaluation / Scores")
    scores = []
    for ai, (env_eval, agent_obj, label) in enumerate(zip(envs, agents, labels)):
        wins = 0
        catches = 0
        total_reward = 0.0
        for e in range(eval_episodes):
            s = env_eval.reset()
            done = False
            steps = 0
            while not done and steps < env_eval.max_steps:
                # agent acts greedily via policy_net
                with torch.no_grad():
                    import torch as _torch
                    stt = _torch.from_numpy(s).float().unsqueeze(0).to(agent_obj.device)
                    q = agent_obj.policy_net(stt)
                    action = int(q.argmax().item())
                s, r, done, info = env_eval.step(action)
                total_reward += r
                steps += 1
            # after episode finished, check outcome
            if hasattr(env_eval, "_runtime_goals") and env_eval.pos in [g['pos'] for g in env_eval._runtime_goals]:
                wins += 1
            if any([d['pos'] == env_eval.pos for d in getattr(env_eval, "_runtime_defenders", [])]):
                catches += 1
        avg_reward = total_reward / eval_episodes
        win_rate = wins / eval_episodes
        catch_rate = catches / eval_episodes
        scores.append({"label": label, "avg_reward": avg_reward, "win_rate": win_rate, "catch_rate": catch_rate})

    # display scores
    score_lines = []
    for sc in scores:
        score_lines.append(f"{sc['label']}: avg_reward={sc['avg_reward']:.3f}, win_rate={sc['win_rate']*100:.1f}%, catch_rate={sc['catch_rate']*100:.1f}%")
    score_area.text("\n".join(score_lines))

    # show final frames
    for i in range(n):
        img_holders[i].image(envs[i].render(mode="rgb_array"), caption=f"{labels[i]} final", use_container_width=True)

    # ALWAYS show defender training curves in a dedicated panel (avoids missing / duplicate traces)
    with st.expander("Defender training curves (per-defender)", expanded=True):
        if any([logs is not None for logs in defender_logs_all]):
            defender_plot_data = {}
            for di, logs in enumerate(defender_logs_all):
                if logs is not None:
                    defender_plot_data[f"defender_{di+1}"] = logs
            if defender_plot_data:
                st.line_chart(defender_plot_data)
        else:
            st.info("No defender training logs available.")

    # ------------------------
    # Hands-on reward editing and export/import (uploader removed as requested)
    # ------------------------
    st.markdown("---")
    st.header("Rewards export / edit")

    if 'last_rewards' in st.session_state:
        lrdata = st.session_state['last_rewards']
        agent_lists = lrdata.get('agents', [])
        defender_lists = lrdata.get('defenders', [])
        max_len = 0
        for a in agent_lists:
            max_len = max(max_len, len(a))
        for d in defender_lists:
            if d is not None:
                max_len = max(max_len, len(d))
        if max_len == 0:
            max_len = 1
        rows = []
        for ep in range(max_len):
            row = {"episode": ep+1}
            for i, a in enumerate(agent_lists):
                row[f"agent_{i+1}"] = a[ep] if ep < len(a) else np.nan
            for j, d in enumerate(defender_lists):
                row[f"defender_{j+1}"] = d[ep] if (d is not None and ep < len(d)) else np.nan
            rows.append(row)
        df = pd.DataFrame(rows)

        st.subheader("Editable reward table (per-episode)")
        # Use experimental data editor if available
        try:
            edited = st.experimental_data_editor(df, num_rows="dynamic")
            if st.button("Save edited rewards to session"):
                st.session_state['edited_rewards_df'] = edited.copy()
                st.success("Edited rewards saved to session state (use 'Apply edited rewards' to override charts/scores).")
        except Exception:
            # fallback: show dataframe and allow download
            st.dataframe(df)
            st.info("Interactive editing not available in this Streamlit version. Download CSV, edit locally and re-upload via your own flow.")

        csv_buf = df.to_csv(index=False).encode('utf-8')
        json_buf = json.dumps({
            "episodes": df['episode'].tolist(),
            "agents": [agent_lists[i] for i in range(len(agent_lists))],
            "defenders": [defender_lists[i] for i in range(len(defender_lists))]
        }, indent=2).encode('utf-8')

        st.download_button("Download rewards CSV", data=csv_buf, file_name="rewards.csv", mime="text/csv")
        st.download_button("Download rewards JSON", data=json_buf, file_name="rewards.json", mime="application/json")

    else:
        st.info("No training rewards available yet. Train agents/defenders then return here.")

else:
    # preview only
    try:
        frame = env.render(mode="rgb_array")
        img_holder.image(frame, caption="Environment", use_container_width=True)
    except Exception as e:
        st.error(f"Render failed: {e}")