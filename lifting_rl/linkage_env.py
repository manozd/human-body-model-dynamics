import csv

import gym
from gym import spaces
import numpy as np
from sympy import lambdify
from scipy.integrate import odeint
from lifting_rl.n_linkage import kane
from scipy import interpolate


class LinkageEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(self, w_params: dict, verbose: bool = False):
        M, F, m_params = kane(n=w_params["N_LINKS"])
        self.M_func = lambdify(m_params, M)
        self.F_func = lambdify(m_params, F)

        high_speed = np.full((w_params["N_LINKS"],), w_params["SPEED_LIMIT"])
        
        
        self.low = np.array([1.19, np.pi/2, 0.2, -1.8, -1.26, -8, -8, -8, -8, -8], dtype=np.float32)
        self.high = np.array([np.pi/2, 2.56, np.pi/2, -1.4, -1, 8, 8, 8, 8, 8], dtype=np.float32)
        low = np.concatenate((w_params["OBS_LOW"], -high_speed))
        high = np.concatenate((w_params["OBS_HIGH"], high_speed))
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        print("observation_space: ", self.observation_space)
        self.action_space = spaces.Box(
            low=-w_params["ACT_LIMIT"],
            high=w_params["ACT_LIMIT"],
            shape=(w_params["N_LINKS"],),
            dtype=np.float32,
        )
        print("action_space: ", self.action_space)

        self.verbose = verbose

        self.time_step = w_params["TIME_STEP"]
        self.nlinks = w_params["N_LINKS"]
        self.act_limit = w_params["ACT_LIMIT"]
        self.speed_limit = w_params["SPEED_LIMIT"]

        self.llength = 0.4
        self.lmass = 1

        self.param_vals = [w_params["PARAM_VALS"][0], 0.4, 1, 0.4, 1, 0.4, 1, 0.4, 1, 0.4, 1]

        self.gpos = None
        self.u = None
        self.viewer = None
        self.reset()

    def reset(self):
        self.state = np.random.uniform(low=self.low, high=self.high)
        self.gpos = np.random.uniform(low=self.low[:self.nlinks], high=self.high[:self.nlinks])
        self.cur_step = 0
        return self._get_obs()

    def inference_reset(self, state, gpos):
        self.state = state
        self.gpos = gpos
        self.cur_step = 0
        return self._get_obs()


    def step(self, u):
        self.u = u * self.act_limit
        t = np.linspace(0, self.time_step, 2)
        next_step = self.cur_step + 1
        state0 = self.state

        if self.verbose:
            print("=" * 50 + "\n")
            print(f"STEP = {self.cur_step}")
            print(f"\t before state: {self.state}")
            print(f"\t before trj: {self.coordinates[self.cur_step]}")
            print(f"\t control = {u}")

        self.state = odeint(self._rhs, state0, t, args=(self.param_vals,))[-1]
        self.state[self.nlinks:] = np.clip(self.state[self.nlinks:], -self.speed_limit, self.speed_limit)
        self.cur_step += 1

        pos_reward = -sum(abs(self.state[:self.nlinks] - self.gpos) ** 2 - abs(self.state[:self.nlinks] - self.gpos)) ** 2
        speed_reward = -sum(abs(self.state[self.nlinks:] / 8))
        control_reward = -sum(abs(self.u / 100))
        goal_reached = sum(abs(self.state[: self.nlinks] - self.gpos[: self.nlinks])) ** 2 < 0.1
        reward = 10*pos_reward

        if self.verbose:
            print(f"\t after state = {self.state}")
            print(f"\t after trj = {self.coordinates[next_step]}")
            print(f"\t is_out_of_bounds = {is_out_of_bounds}")
            print(f"\t reward = {reward}")
            print("=" * 50 + "\n")

        is_end = self.cur_step == 288

        terminate = self._is_out_of_bounds() or is_end
        if self._is_out_of_bounds():
            reward = -5000
        
        if goal_reached:
            reward += 100

        if terminate and self.verbose:
            print("*" * 50)
            print("*****" + "TERMINATE" + "*****")
            print("*" * 50)
        return self._get_obs(), reward, terminate, {}

    def _is_out_of_bounds(self):
        return not self.observation_space.contains(self.state)

    def _normalize(self, vector, max):
        return np.array(
            [coord / np.sqrt(sum(vector ** 2)) for coord in vector], dtype=np.float32
        )

    def _get_obs(self):
        coords = self.state[:self.nlinks]
        speed = self.state[self.nlinks:]
        return np.hstack((np.sin(coords), np.cos(coords), np.sin(self.gpos), np.cos(self.gpos), speed))

    def _rhs(self, x, t, args):
        arguments = np.hstack((x, self.u, args))
        dx = np.array(
            np.linalg.solve(self.M_func(*arguments), self.F_func(*arguments))
        ).T[0]
        return dx

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(600, 400)
            bound = 1.5  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -0.5, bound)

        lpoints = [[0, 0]]
        for i in range(self.nlinks):
            pcos = lpoints[-1][0] + self.llength * np.cos(self.state[i])
            psin = lpoints[-1][1] + self.llength * np.sin(self.state[i])
            lpoints.append([pcos, psin])

        gpoints = [[0, 0]]
        for i in range(self.nlinks):
            pcos = gpoints[-1][0] + self.llength * np.cos(self.gpos[i])
            psin = gpoints[-1][1] + self.llength * np.sin(self.gpos[i])
            gpoints.append([pcos, psin])

        llengths = [self.llength for i in range(self.nlinks)]

        lthetas = self.state[:self.nlinks] % (2*np.pi)
        gthetas = self.gpos % (2*np.pi)


        for ((x, y), th, llen) in zip(gpoints, gthetas, llengths):
            l, r, t, b = 0, llen, 0.02, -0.02
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0.8, 0.8, 0.8)
            circ = self.viewer.draw_circle(0.02)
            circ.set_color(0.0, 0.0, 0)
            circ.add_attr(jtransform)

        for ((x, y), th, llen) in zip(lpoints, lthetas, llengths):
            l, r, t, b = 0, llen, 0.04, -0.04
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0.8, 0.3, 0.3)
            circ = self.viewer.draw_circle(0.04)
            circ.set_color(0.0, 0.0, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
