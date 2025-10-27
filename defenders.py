# defenders.py
import random
import numpy as np
from collections import defaultdict, deque

import torch
from dqn import DQNAgent

# Helper wrappers to give a uniform select_action(def_pos, agent_pos, env) interface
class TabularPolicyWrapper:
    def __init__(self, tabular_agent):
        # tabular_agent is instance of TabularAgent used in training
        self.agent = tabular_agent

    def select_action(self, def_pos, agent_pos, env):
        # state tuple expected by TabularAgent: (dx,dy,ax,ay)
        s = (def_pos[0], def_pos[1], agent_pos[0], agent_pos[1])
        return int(self.agent.select_action(s))

class DQNPolicyWrapper:
    def __init__(self, dqn_agent):
        self.agent = dqn_agent
        # set epsilon to 0 for deterministic evaluation, but we will call policy_net directly in select_action
    def select_action(self, def_pos, agent_pos, env):
        s = np.array([def_pos[0]/(env.size-1), def_pos[1]/(env.size-1), agent_pos[0]/(env.size-1), agent_pos[1]/(env.size-1)], dtype=np.float32)
        # use policy_net deterministically
        with torch.no_grad():
            import torch as _torch
            st = _torch.from_numpy(s).float().unsqueeze(0).to(self.agent.device)
            q = self.agent.policy_net(st)
            return int(q.argmax().item())

# Small tabular agent class reused from earlier code
class TabularAgent:
    def __init__(self, grid_size, action_dim, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=500):
        self.size = grid_size
        self.actions = list(range(action_dim))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        self.Q = defaultdict(lambda: np.zeros(len(self.actions)))
        self.returns = defaultdict(list)

    def _eps(self):
        e = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1.0 * self.steps / max(1, self.epsilon_decay))
        return e

    def select_action(self, state):
        self.steps += 1
        if random.random() < self._eps():
            return random.choice(self.actions)
        q = self.Q[state]
        return int(np.argmax(q))

    def update_q_learning(self, s, a, r, s2, done):
        key = s
        next_key = s2
        best_next = 0.0 if done else np.max(self.Q[next_key])
        self.Q[key][a] += self.alpha * (r + self.gamma * best_next - self.Q[key][a])

    def update_sarsa(self, s, a, r, s2, a2, done):
        key = s
        next_key = s2
        q_next = 0.0 if done else self.Q[next_key][a2]
        self.Q[key][a] += self.alpha * (r + self.gamma * q_next - self.Q[key][a])

    def update_monte_carlo_episode(self, episode):
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = self.gamma * G + r
            key = s
            if (key, a) in visited:
                continue
            visited.add((key, a))
            self.returns[(key,a)].append(G)
            self.Q[key][a] = np.mean(self.returns[(key,a)])

    def update_td0(self, s, r, s2, done):
        key = s
        next_key = s2
        v = np.max(self.Q[key])
        vnext = 0.0 if done else np.max(self.Q[next_key])
        delta = r + self.gamma * vnext - v
        self.Q[key] += self.alpha * delta / max(1, len(self.actions))

def _state_from_env(env, defender_index=0):
    # helper to get def_pos and agent_pos
    if not hasattr(env, "_runtime_defenders") or len(env._runtime_defenders) == 0:
        return None
    dpos = env._runtime_defenders[defender_index]['pos']
    apos = env.pos
    return (dpos[0], dpos[1], apos[0], apos[1])

def sample_random_agent_action(env):
    return env.action_space.sample()

def _find_free_cell(env):
    cells = [(x, y) for x in range(env.size) for y in range(env.size)]
    forbidden = {tuple(env.start)}
    if hasattr(env, "_runtime_obstacles"):
        forbidden |= set([o['pos'] for o in env._runtime_obstacles])
    if hasattr(env, "_runtime_goals"):
        forbidden |= set([g['pos'] for g in env._runtime_goals])
    if hasattr(env, "_runtime_defenders"):
        forbidden |= set([d['pos'] for d in env._runtime_defenders])
    avail = [c for c in cells if c not in forbidden]
    if not avail:
        avail = [c for c in cells if c != tuple(env.start)]
    return random.choice(avail) if avail else tuple(env.start)

def train_defender_tabular(
    env,
    defender_index=0,
    episodes=200,
    max_steps=50,
    algo="q_learning",
    alpha=0.2,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=500,
    render_every=50,
):
    """
    Train a single defender (tabular variants). Returns (policy_wrapper, logs).
    If no defender exists in env, a temporary one is added for training.
    """
    # ensure runtime defenders exist
    env.reset()
    if not getattr(env, "defenders", None) or len(env.defenders) == 0:
        pos = _find_free_cell(env)
        env.defenders = [{'pos': pos, 'moving': False, 'policy': None}]
        env.reset()

    # create TabularAgent
    size = env.size
    action_dim = env.action_space.n
    tab = TabularAgent(grid_size=size, action_dim=action_dim, alpha=alpha, gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay)
    logs = []
    for ep in range(1, episodes+1):
        env.reset()
        ep_reward = 0.0
        episode_traj = []
        for t in range(max_steps):
            # defender state
            if defender_index >= len(env._runtime_defenders):
                di = max(0, len(env._runtime_defenders)-1)
            else:
                di = defender_index
            dpos = env._runtime_defenders[di]['pos']
            apos = env.pos
            s = (dpos[0], dpos[1], apos[0], apos[1])
            a = tab.select_action(s)
            # attempt defender move
            old = env._runtime_defenders[di]['pos']
            x,y = old
            if a == 0 and y > 0:
                ny = y - 1; nx = x
            elif a == 1 and x < env.size - 1:
                nx = x + 1; ny = y
            elif a == 2 and y < env.size - 1:
                ny = y + 1; nx = x
            elif a == 3 and x > 0:
                nx = x - 1; ny = y
            else:
                nx, ny = x, y
            forbidden = set([o['pos'] for o in env._runtime_obstacles]) | set([g['pos'] for g in env._runtime_goals])
            other_defs = set([d['pos'] for d in env._runtime_defenders]) - {old}
            if (nx, ny) not in forbidden and (nx, ny) not in other_defs:
                env._runtime_defenders[di]['pos'] = (nx, ny)
            # prey (agent) moves randomly for training
            prey_action = sample_random_agent_action(env)
            _, _, done, _ = env.step(prey_action)
            dpos2 = env._runtime_defenders[di]['pos']
            apos2 = env.pos
            if dpos2 == apos2:
                r = 1.0
                done = True
            else:
                r = -0.01
            ep_reward += r
            s2 = (dpos2[0], dpos2[1], apos2[0], apos2[1])
            if algo == "q_learning":
                tab.update_q_learning(s, a, r, s2, done)
            elif algo == "sarsa":
                a2 = tab.select_action(s2)
                tab.update_sarsa(s, a, r, s2, a2, done)
            elif algo == "monte_carlo":
                episode_traj.append((s,a,r))
            elif algo == "td0":
                tab.update_td0(s, r, s2, done)
            if done:
                break
        if algo == "monte_carlo":
            tab.update_monte_carlo_episode(episode_traj)
        logs.append(ep_reward)

    # return a wrapper that uses the tabular policy for runtime
    wrapper = TabularPolicyWrapper(tab)
    return wrapper, logs

def train_defender_dqn(
    env,
    defender_index=0,
    episodes=200,
    max_steps=50,
    lr=1e-3,
    gamma=0.99,
    batch_size=32,
    replay_size=2000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=500,
    target_update=50,
    device=None,
    render_every=50
):
    """
    Train defender using a DQNAgent. Returns (policy_wrapper, logs).
    """
    device = device or torch.device("cpu")
    env.reset()
    if not getattr(env, "defenders", None) or len(env.defenders) == 0:
        pos = _find_free_cell(env)
        env.defenders = [{'pos': pos, 'moving': False, 'policy': None}]
        env.reset()

    agent = DQNAgent(state_dim=4, action_dim=env.action_space.n, lr=lr, gamma=gamma, batch_size=batch_size, replay_size=replay_size, device=device, epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay, variant="deepmind", target_update_freq=target_update)
    logs = []
    for ep in range(1, episodes+1):
        env.reset()
        ep_reward = 0.0
        for t in range(max_steps):
            if defender_index >= len(env._runtime_defenders):
                di = max(0, len(env._runtime_defenders)-1)
            else:
                di = defender_index
            dpos = env._runtime_defenders[di]['pos']
            apos = env.pos
            s = np.array([dpos[0]/(env.size-1), dpos[1]/(env.size-1), apos[0]/(env.size-1), apos[1]/(env.size-1)], dtype=np.float32)
            a = agent.select_action(s)
            # apply defender move
            old = env._runtime_defenders[di]['pos']
            x,y = old
            if a == 0 and y > 0:
                ny = y - 1; nx = x
            elif a == 1 and x < env.size - 1:
                nx = x + 1; ny = y
            elif a == 2 and y < env.size - 1:
                ny = y + 1; nx = x
            elif a == 3 and x > 0:
                nx = x - 1; ny = y
            else:
                nx, ny = x, y
            forbidden = set([o['pos'] for o in env._runtime_obstacles]) | set([g['pos'] for g in env._runtime_goals])
            other_defs = set([d['pos'] for d in env._runtime_defenders]) - {old}
            if (nx, ny) not in forbidden and (nx, ny) not in other_defs:
                env._runtime_defenders[di]['pos'] = (nx, ny)
            # prey moves randomly for training
            prey_action = sample_random_agent_action(env)
            _, _, done, _ = env.step(prey_action)
            dpos2 = env._runtime_defenders[di]['pos']
            apos2 = env.pos
            if dpos2 == apos2:
                r = 1.0
                done = True
            else:
                r = -0.01
            s2 = np.array([dpos2[0]/(env.size-1), dpos2[1]/(env.size-1), apos2[0]/(env.size-1), apos2[1]/(env.size-1)], dtype=np.float32)
            agent.push(s, a, r, s2, done)
            loss = agent.update()
            ep_reward += r
            if done:
                break
        logs.append(ep_reward)

    wrapper = DQNPolicyWrapper(agent)
    return wrapper, logs