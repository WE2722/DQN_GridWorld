# train.py
import numpy as np
import torch
from dqn import DQNAgent
from collections import deque

def train_generator(
    env,
    agent: DQNAgent,
    episodes=200,
    max_steps_per_episode=50,
    render_every=1,
    update_every=1,
    yield_every_episode=1,
):
    """
    Generator that yields after each yield_every_episode with a dict:
    {
      'episode': int,
      'episode_reward': float,
      'frame': np.array (RGB),
      'mean_reward': float,
      'loss': float or None,
      'total_steps': int
    }
    """
    rewards_history = []
    recent = deque(maxlen=100)
    loss = None
    total_steps = 0
    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        for t in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, next_state, done)
            # update network every step (small env, so it's fine) or every update_every
            if (t + 1) % update_every == 0:
                loss = agent.update()
            state = next_state
            ep_reward += reward
            total_steps += 1
            if done:
                break

        rewards_history.append(ep_reward)
        recent.append(ep_reward)
        mean_reward = float(np.mean(recent)) if recent else 0.0

        # optionally render
        frame = env.render(mode="rgb_array") if (ep % render_every == 0 or ep == episodes) else None

        if ep % yield_every_episode == 0:
            yield {
                'episode': ep,
                'episode_reward': float(ep_reward),
                'frame': frame,
                'mean_reward': float(mean_reward),
                'loss': loss,
                'total_steps': total_steps
            }

def compare_train_generator(
    envs,
    agents,
    episodes=200,
    max_steps_per_episode=50,
    render_every=1,
    update_every=1,
    yield_every_episode=1,
):
    """
    Generator to train multiple agents on separate but identically-configured envs and yield per-agent info.

    envs: list of gym.Env (one per agent)
    agents: list of DQNAgent (one per agent)
    Yields:
    {
      'episode': int,
      'agents': [
         {
           'name': str,
           'episode_reward': float,
           'frame': np.array or None,
           'mean_reward': float,
           'loss': float or None,
           'total_steps': int
         }, ...
      ]
    }
    """
    n = len(agents)
    rewards_history = [[] for _ in range(n)]
    recents = [deque(maxlen=100) for _ in range(n)]
    losses = [None] * n
    total_steps = [0] * n

    for ep in range(1, episodes + 1):
        agents_data = []
        for i, (env, agent) in enumerate(zip(envs, agents)):
            state = env.reset()
            ep_reward = 0.0
            loss = None
            for t in range(max_steps_per_episode):
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.push(state, action, reward, next_state, done)
                if (t + 1) % update_every == 0:
                    loss = agent.update()
                state = next_state
                ep_reward += reward
                total_steps[i] += 1
                if done:
                    break

            rewards_history[i].append(ep_reward)
            recents[i].append(ep_reward)
            mean_reward = float(np.mean(recents[i])) if recents[i] else 0.0
            frame = env.render(mode="rgb_array") if (ep % render_every == 0 or ep == episodes) else None
            losses[i] = loss

            agents_data.append({
                'name': 'agent_{}'.format(i),
                'episode_reward': float(ep_reward),
                'frame': frame,
                'mean_reward': float(mean_reward),
                'loss': loss,
                'total_steps': total_steps[i]
            })

        if ep % yield_every_episode == 0:
            yield {
                'episode': ep,
                'agents': agents_data
            }