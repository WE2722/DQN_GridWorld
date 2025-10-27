# grid_env.py
import gym
from gym import spaces
import numpy as np
import random
from copy import deepcopy

class GridWorldEnv(gym.Env):
    """
    GridWorld with support for multiple goals, obstacles and defender agents.
    - If provided_* lists are not None they will be used verbatim (even if empty).
    - Defenders may have a 'policy' attached (object with select_action(def_pos, agent_pos, env) -> action)
      In that case the defender will act using that policy each step.
    - Goals / obstacles / defenders can be marked as moving (random walk) or have policies (for defenders).
    - Reward parameters are configurable via constructor:
        reward_goal, reward_defender, reward_obstacle, reward_step
    """
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        size=6,
        start=(0,0),
        goals=None,
        num_goals=1,
        obstacles=None,
        num_obstacles=5,
        defenders=None,
        num_defenders=0,
        defenders_move=False,
        moving_goals=False,
        moving_obstacles=False,
        max_steps=50,
        seed=None,
        reward_goal: float = 1.0,
        reward_defender: float = -1.0,
        reward_obstacle: float = -1.0,
        reward_step: float = -0.01,
    ):
        super().__init__()
        self.size = size
        self.start = tuple(start)
        self.max_steps = max_steps
        self._seed_val = seed

        # reward configuration
        self.reward_goal = float(reward_goal)
        self.reward_defender = float(reward_defender)
        self.reward_obstacle = float(reward_obstacle)
        self.reward_step = float(reward_step)

        # preserve the "provided" objects exactly when set (even empty list)
        self._provided_goals = goals
        self._provided_obstacles = obstacles
        self._provided_defenders = defenders

        self.num_goals = num_goals
        self.num_obstacles = num_obstacles
        self.num_defenders = num_defenders
        self.defenders_move_flag = defenders_move
        self.moving_goals_flag = moving_goals
        self.moving_obstacles_flag = moving_obstacles

        # actions: up, right, down, left
        self.action_space = spaces.Discrete(4)
        # observation: agent normalized x,y
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self.seed(seed)
        self._init_layout()
        self.reset()

    def seed(self, s=None):
        self._seed_val = s
        np.random.seed(s)
        random.seed(s)

    def _init_layout(self):
        cells = [(x, y) for x in range(self.size) for y in range(self.size)]
        forbidden = {tuple(self.start)}

        # goals: use provided if not None, else sample num_goals
        if self._provided_goals is not None:
            self.goals = []
            for g in self._provided_goals:
                if isinstance(g, dict):
                    pos = tuple(g.get('pos'))
                    moving = bool(g.get('moving', False))
                else:
                    pos = tuple(g)
                    moving = self.moving_goals_flag
                self.goals.append({'pos': pos, 'moving': moving})
                forbidden.add(pos)
        else:
            self.goals = []
            avail = [c for c in cells if c not in forbidden]
            random.shuffle(avail)
            for _ in range(self.num_goals):
                if not avail:
                    break
                g = avail.pop()
                self.goals.append({'pos': g, 'moving': self.moving_goals_flag})
                forbidden.add(g)

        # obstacles: use provided if not None, else sample num_obstacles
        if self._provided_obstacles is not None:
            self.obstacles = []
            for o in self._provided_obstacles:
                if isinstance(o, dict):
                    pos = tuple(o.get('pos'))
                    moving = bool(o.get('moving', False))
                else:
                    pos = tuple(o)
                    moving = self.moving_obstacles_flag
                self.obstacles.append({'pos': pos, 'moving': moving})
                forbidden.add(pos)
        else:
            self.obstacles = []
            avail = [c for c in cells if c not in forbidden]
            random.shuffle(avail)
            for _ in range(self.num_obstacles):
                if not avail:
                    break
                o = avail.pop()
                self.obstacles.append({'pos': o, 'moving': self.moving_obstacles_flag})
                forbidden.add(o)

        # defenders: use provided if not None, else sample num_defenders
        if self._provided_defenders is not None:
            self.defenders = []
            for d in self._provided_defenders:
                pos = tuple(d.get('pos')) if isinstance(d, dict) else tuple(d)
                moving = bool(d.get('moving', False)) if isinstance(d, dict) else self.defenders_move_flag
                policy = d.get('policy') if isinstance(d, dict) else None
                self.defenders.append({'pos': pos, 'moving': moving, 'policy': policy})
                forbidden.add(pos)
        else:
            self.defenders = []
            avail = [c for c in cells if c not in forbidden]
            random.shuffle(avail)
            for _ in range(self.num_defenders):
                if not avail:
                    break
                p = avail.pop()
                self.defenders.append({'pos': p, 'moving': self.defenders_move_flag, 'policy': None})
                forbidden.add(p)

    def reset(self):
        self.pos = tuple(self.start)
        self.steps = 0
        self.done = False
        # runtime copies: keep same length and metadata; policies may be attached later
        self._runtime_goals = [{'pos': g['pos'], 'moving': g['moving']} for g in self.goals]
        self._runtime_obstacles = [{'pos': o['pos'], 'moving': o['moving']} for o in self.obstacles]
        # copy policy references so they survive reset only if they are in self.defenders
        self._runtime_defenders = [{'pos': d['pos'], 'moving': d.get('moving', False), 'policy': d.get('policy', None)} for d in self.defenders]
        return self._get_obs()

    def _get_obs(self):
        x, y = self.pos
        return np.array([x / (self.size - 1), y / (self.size - 1)], dtype=np.float32)

    def _in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def _entity_positions(self, entities):
        return [e['pos'] for e in entities]

    def _move_entity_list(self, entities, forbidden_targets, use_policy=False):
        """
        Generic entity mover:
        - if entity has 'policy' and use_policy True: ask policy for action
        - else if entity['moving'] True: random move
        - forbidden_targets: positions this entity must not move into (set)
        """
        new_positions = set(self._entity_positions(entities))
        for idx, e in enumerate(entities):
            if use_policy and e.get('policy') is not None:
                # policy is expected to be an object with select_action(def_pos, agent_pos, env) -> action (0..3)
                try:
                    def_pos = e['pos']
                    agent_pos = self.pos
                    action = e['policy'].select_action(def_pos, agent_pos, self)
                except Exception:
                    action = None
                if action is None:
                    candidates = [(e['pos'][0], e['pos'][1])]
                else:
                    x, y = e['pos']
                    if action == 0:
                        candidates = [(x, y-1), (x, y)]
                    elif action == 1:
                        candidates = [(x+1, y), (x, y)]
                    elif action == 2:
                        candidates = [(x, y+1), (x, y)]
                    elif action == 3:
                        candidates = [(x-1, y), (x, y)]
                    else:
                        candidates = [(e['pos'][0], e['pos'][1])]
            else:
                if not e.get('moving', False):
                    candidates = [(e['pos'][0], e['pos'][1])]
                else:
                    x, y = e['pos']
                    candidates = [(x, y), (x, y-1), (x+1, y), (x, y+1), (x-1, y)]
                    random.shuffle(candidates)

            moved = False
            for nx, ny in candidates:
                if not self._in_bounds(nx, ny):
                    continue
                if (nx, ny) in forbidden_targets:
                    continue
                other_positions = new_positions - {e['pos']}
                if (nx, ny) in other_positions:
                    continue
                # accept move
                new_positions.discard(e['pos'])
                new_positions.add((nx, ny))
                e['pos'] = (nx, ny)
                moved = True
                break
            if not moved:
                pass

    def _move_goals(self):
        forbidden = set(self._entity_positions(self._runtime_obstacles)) | set(self._entity_positions(self._runtime_defenders)) | {self.start}
        self._move_entity_list(self._runtime_goals, forbidden, use_policy=False)

    def _move_obstacles(self):
        forbidden = set(self._entity_positions(self._runtime_goals)) | set(self._entity_positions(self._runtime_defenders)) | {self.start}
        self._move_entity_list(self._runtime_obstacles, forbidden, use_policy=False)

    def _move_defenders(self):
        forbidden = set(self._entity_positions(self._runtime_goals)) | set(self._entity_positions(self._runtime_obstacles)) | {self.start}
        self._move_entity_list(self._runtime_defenders, forbidden, use_policy=True)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {}

        # Move moving goals/obstacles/defenders BEFORE agent acts
        if any(g['moving'] for g in self._runtime_goals):
            self._move_goals()
        if any(o['moving'] for o in self._runtime_obstacles):
            self._move_obstacles()
        if any(d['moving'] or d.get('policy') for d in self._runtime_defenders):
            self._move_defenders()

        # collisions from moved entities onto agent
        if self.pos in self._entity_positions(self._runtime_obstacles):
            self.done = True
            return self._get_obs(), float(self.reward_obstacle), True, {"reason": "obstacle_moved_on_agent"}
        if self.pos in self._entity_positions(self._runtime_defenders):
            self.done = True
            return self._get_obs(), float(self.reward_defender), True, {"reason": "defender_moved_on_agent"}
        if self.pos in self._entity_positions(self._runtime_goals):
            self.done = True
            return self._get_obs(), float(self.reward_goal), True, {"reason": "goal_moved_on_agent"}

        # Agent action
        x, y = self.pos
        if action == 0 and y > 0:
            ny = y - 1; nx = x
        elif action == 1 and x < self.size - 1:
            nx = x + 1; ny = y
        elif action == 2 and y < self.size - 1:
            ny = y + 1; nx = x
        elif action == 3 and x > 0:
            nx = x - 1; ny = y
        else:
            nx, ny = x, y

        # Can't move into runtime obstacles
        if (nx, ny) not in self._entity_positions(self._runtime_obstacles):
            self.pos = (nx, ny)

        self.steps += 1
        reward = float(self.reward_step)

        # collisions (agent moved into defender/obstacle)
        if self.pos in self._entity_positions(self._runtime_defenders):
            self.done = True
            reward = float(self.reward_defender)
            return self._get_obs(), float(reward), True, {"reason": "agent_hit_defender"}
        if self.pos in self._entity_positions(self._runtime_obstacles):
            self.done = True
            reward = float(self.reward_obstacle)
            return self._get_obs(), float(reward), True, {"reason": "agent_hit_obstacle"}

        # goal check
        if self.pos in self._entity_positions(self._runtime_goals):
            self.done = True
            reward = float(self.reward_goal)
            return self._get_obs(), float(reward), True, {"reason": "goal_reached"}

        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), float(reward), bool(self.done), {}

    def render(self, mode="rgb_array"):
        cell = 40
        img_size = self.size * cell
        canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

        # grid lines
        for i in range(1, self.size):
            canvas[i*cell-1:i*cell+1, :, :] = 200
            canvas[:, i*cell-1:i*cell+1, :] = 200

        # obstacles (dark grey)
        for o in getattr(self, "_runtime_obstacles", self.obstacles):
            ox, oy = o['pos'] if isinstance(o, dict) else o
            x0, y0 = ox*cell, oy*cell
            canvas[y0+2:y0+cell-2, x0+2:x0+cell-2, :] = np.array([110, 110, 110], dtype=np.uint8)

        # goals (green)
        for g in getattr(self, "_runtime_goals", self.goals):
            gx, gy = g['pos'] if isinstance(g, dict) else g
            x0, y0 = gx*cell, gy*cell
            canvas[y0+4:y0+cell-4, x0+4:x0+cell-4, :] = np.array([50, 200, 50], dtype=np.uint8)

        # defenders (red)
        for d in getattr(self, "_runtime_defenders", self.defenders):
            dx, dy = d['pos'] if isinstance(d, dict) else d
            x0, y0 = dx*cell, dy*cell
            canvas[y0+6:y0+cell-6, x0+6:x0+cell-6, :] = np.array([200, 50, 50], dtype=np.uint8)

        # agent (blue)
        ax, ay = self.pos
        x0, y0 = ax*cell, ay*cell
        canvas[y0+6:y0+cell-6, x0+6:x0+cell-6, :] = np.array([50, 50, 200], dtype=np.uint8)

        return canvas