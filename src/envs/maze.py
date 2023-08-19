"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
from envs.multiagentenv import MultiAgentEnv
import torch as th
import pygame
from utils.dict2namedtuple import convert
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width


class Maze(MultiAgentEnv, tk.Tk, object):
    def __init__(self, **kwargs):
        super(Maze, self).__init__()
        
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args


        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

        self.n_agents = args.n_agents
        self.episode_limit = args.episode_limit




    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 3])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.hell1)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return reward, done, s_

    def render(self):
        # time.sleep(0.01)
        self.update()



    # ---------- OBSERVATION METHODS -----------------------------------------------------------------------------------
    def get_obs_agent(self, agent_id, batch=0):
        obs, _ = self._observe([agent_id])
        # If the frozen agents are removed, their observation is blank
        if self.capture_freezes and self.remove_frozen and self.agents_not_frozen[agent_id, batch] == 0:
            obs *= 0
        return obs

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        # Return the entire grid
        if self.batch_mode:
            return self.grid.copy().reshape(self.state_size)
        else:
            return self.grid[0, :, :, :].reshape(self.state_size)

    # ---------- GETTERS -----------------------------------------------------------------------------------------------
    def get_total_actions(self):
        return self.n_actions

    def get_avail_agent_actions(self, agent_id):
        """ Currently runs only with batch_size==1. """
        if self.agents_not_frozen[agent_id] == 0:
            # All agents that are frozen can only perform the "stay" action
            avail_actions = [0 for _ in range(self.n_actions)]
            avail_actions[self.action_labels['stay']] = 1
        elif self.toroidal:
            # In a toroidal environment, all movement actions are allowed
            avail_actions = [1 for _ in range(self.n_actions)]
        else:
            # In a bounded environment, you cannot move into walls
            new_pos = self.agents[agent_id, 0, :] + self.actions[:self.n_actions]
            allowed = np.logical_and(new_pos >= 0, new_pos < self.grid_shape).all(axis=1)
            assert np.any(allowed), "No available action in the environment: this should never happen!"
            avail_actions = [int(allowed[a]) for a in range(self.n_actions)]
        # If the agent is not frozen, the 'catch' action is only available next to a prey
        if self.capture_action and self.agents_not_frozen[agent_id] > 0:
            avail_actions[self.action_labels['catch']] = 0
            # Check with virtual move actions if there is a prey next to the agent
            possible_catches = range(4) if not self.directed_observations \
                else range(self.agents_orientation[agent_id, 0], self.agents_orientation[agent_id, 0] + 1)
            for u in possible_catches:
                if self._move_actor(self.agents[agent_id, 0, :], u, 0, np.asarray([1, 2], dtype=int_type))[1]:
                    avail_actions[self.action_labels['catch']] = 1
                    break
        return avail_actions

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return avail_actions

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        return self.state_size

    def get_stats(self):
        pass

    def get_env_info(self):
        info = MultiAgentEnv.get_env_info(self)
        return info

    # --------- RENDER METHODS -----------------------------------------------------------------------------------------
    def close(self):
        if self.made_screen:
            pygame.quit()
        print("Closing Multi-Agent Navigation")

    def render_array(self):
        # Return an rgb array of the frame. Not implemented!
        return None