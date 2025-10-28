"""Tests for GridWorld environment"""
import pytest
import numpy as np
from grid_env import GridWorldEnv


def test_gridworld_creation():
    """Test that GridWorld environment can be created"""
    env = GridWorldEnv(size=5, num_goals=1, num_obstacles=2)
    assert env is not None
    assert env.size == 5


def test_gridworld_reset():
    """Test that environment can be reset"""
    env = GridWorldEnv(size=5, num_goals=1, num_obstacles=2)
    obs = env.reset()
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)  # (x, y) position


def test_gridworld_step():
    """Test that environment can perform a step"""
    env = GridWorldEnv(size=5, num_goals=1, num_obstacles=2)
    env.reset()
    obs, reward, done, info = env.step(0)  # Move up
    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_gridworld_actions():
    """Test that all actions are valid"""
    env = GridWorldEnv(size=5, num_goals=1, num_obstacles=2)
    env.reset()
    # Test all 4 actions (up, down, left, right)
    for action in range(4):
        obs, reward, done, info = env.step(action)
        assert obs is not None


def test_gridworld_bounds():
    """Test that agent stays within grid bounds"""
    env = GridWorldEnv(size=5, num_goals=1, num_obstacles=0)
    env.reset()
    # Check agent position is within bounds
    assert env.pos[0] >= 0
    assert env.pos[0] < env.size
    assert env.pos[1] >= 0
    assert env.pos[1] < env.size
