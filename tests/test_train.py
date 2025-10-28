"""Tests for training functions"""
import pytest
import numpy as np
from grid_env import GridWorldEnv
from dqn import DQNAgent
from train import train_generator


def test_train_generator_exists():
    """Test that train_generator function exists and is callable"""
    assert callable(train_generator)


def test_train_generator_runs():
    """Test that training generator can run for a few episodes"""
    env = GridWorldEnv(size=5, num_goals=1, num_obstacles=2)
    agent = DQNAgent(
        state_dim=2,
        action_dim=4,
        variant="vanilla",
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500
    )
    
    # Run training for just 2 episodes to test it works
    episode_count = 0
    for result in train_generator(
        env=env,
        agent=agent,
        episodes=2,
        render_every=100
    ):
        episode_count += 1
        assert 'episode' in result
        assert 'episode_reward' in result
        if episode_count >= 2:
            break
    
    assert episode_count == 2


def test_training_produces_rewards():
    """Test that training produces reward values"""
    env = GridWorldEnv(size=5, num_goals=1, num_obstacles=1)
    agent = DQNAgent(
        state_dim=2,
        action_dim=4,
        variant="vanilla",
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500
    )
    
    rewards = []
    for result in train_generator(
        env=env,
        agent=agent,
        episodes=3,
        render_every=100
    ):
        rewards.append(result['episode_reward'])
    
    assert len(rewards) == 3
    assert all(isinstance(r, (int, float)) for r in rewards)
