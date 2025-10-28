"""Tests for DQN agent"""
import pytest
import torch
import numpy as np
from dqn import QNetwork, ReplayBuffer, DQNAgent


def test_qnetwork_creation():
    """Test that Q-Network can be created"""
    network = QNetwork(input_dim=2, hidden=64, outputs=4)
    assert network is not None
    assert isinstance(network, torch.nn.Module)


def test_qnetwork_forward():
    """Test forward pass of Q-Network"""
    network = QNetwork(input_dim=2, hidden=64, outputs=4)
    state = torch.FloatTensor([[0.5, 0.5]])
    q_values = network(state)
    assert q_values.shape == (1, 4)
    assert not torch.isnan(q_values).any()


def test_replay_buffer_creation():
    """Test that replay buffer can be created"""
    buffer = ReplayBuffer(capacity=1000)
    assert buffer is not None
    assert len(buffer) == 0


def test_replay_buffer_push():
    """Test adding experiences to replay buffer"""
    buffer = ReplayBuffer(capacity=1000)
    state = np.array([0.5, 0.5])
    action = 0
    reward = 1.0
    next_state = np.array([0.6, 0.5])
    done = False
    
    buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 1


def test_replay_buffer_sample():
    """Test sampling from replay buffer"""
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some experiences
    for i in range(100):
        state = np.array([i * 0.01, i * 0.01])
        buffer.push(state, 0, 1.0, state, False)
    
    # Sample a batch
    batch = buffer.sample(32)
    # batch is a Transition named tuple with 5 fields
    assert len(batch) == 5
    states, actions, rewards, next_states, dones = batch
    # Each field is a tuple of 32 items
    assert len(states) == 32
    assert len(actions) == 32


def test_dqn_agent_creation():
    """Test that DQN agent can be created"""
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
    assert agent is not None


def test_dqn_agent_select_action():
    """Test action selection"""
    agent = DQNAgent(
        state_dim=2,
        action_dim=4,
        variant="vanilla",
        lr=0.001,
        gamma=0.99,
        epsilon_start=0.0,  # No exploration for deterministic test
        epsilon_end=0.0,
        epsilon_decay=1.0
    )
    state = np.array([0.5, 0.5])
    action = agent.select_action(state)
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < 4


def test_dqn_agent_variants():
    """Test that all DQN algorithm variants can be created"""
    for variant in ["vanilla", "deepmind", "double"]:
        agent = DQNAgent(
            state_dim=2,
            action_dim=4,
            variant=variant,
            lr=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=500
        )
        assert agent is not None
        assert agent.variant == variant
