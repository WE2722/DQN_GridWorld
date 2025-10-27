# dqn.py
import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden=64, outputs=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, outputs)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        batch_size=32,
        replay_size=10000,
        device=None,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=500,
        variant="deepmind",
        target_update_freq=100
    ):
        """
        variant: one of "vanilla", "deepmind", "double"
        - "vanilla": no separate target net, targets computed using policy_net
        - "deepmind": uses target_net for max-next-Q
        - "double": Double DQN: action is selected by policy_net, evaluated by target_net
        """
        self.device = device or torch.device("cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(replay_size)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        assert variant in ("vanilla", "deepmind", "double")
        self.variant = variant
        self.target_update_freq = target_update_freq

        self.policy_net = QNetwork(state_dim, hidden=64, outputs=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        if self.variant in ("deepmind", "double"):
            self.target_net = QNetwork(state_dim, hidden=64, outputs=action_dim).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        else:
            self.target_net = None

        self.loss_fn = nn.MSELoss()
        self.total_updates = 0

    def select_action(self, state):
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
              np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q = self.policy_net(s)
                return int(q.argmax().item())

    def push(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None

        transitions = self.replay.sample(self.batch_size)
        states = torch.tensor(np.array(transitions.state), dtype=torch.float32).to(self.device)
        actions = torch.tensor(transitions.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(transitions.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(transitions.next_state), dtype=torch.float32).to(self.device)
        dones = torch.tensor(transitions.done, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        # target calculation by variant
        with torch.no_grad():
            if self.variant == "deepmind" and self.target_net is not None:
                next_q = self.target_net(next_states)
                max_next_q = next_q.max(1)[0].unsqueeze(1)
                target = rewards + (1.0 - dones) * self.gamma * max_next_q
            elif self.variant == "double" and self.target_net is not None:
                # Double DQN: action selection by policy_net, evaluation by target_net
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)  # shape (batch,1)
                next_q_target = self.target_net(next_states).gather(1, next_actions)
                target = rewards + (1.0 - dones) * self.gamma * next_q_target
            else:
                # vanilla: use policy_net for next-state max
                next_q = self.policy_net(next_states)
                max_next_q = next_q.max(1)[0].unsqueeze(1)
                target = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.total_updates += 1
        if self.target_net is not None and (self.total_updates % self.target_update_freq == 0):
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())