from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium import register
from ray.rllib.env.multi_agent_env import  MultiAgentEnv


class Env_0(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()
        num_agents = 2
        # define chip
        self.height = 10
        self.width = 10
        self.channel = 1  # RGB 图像
        self.chip =  np.zeros((10, 10), dtype=np.uint8)
        self.chip[0][0] = 1
        self.chip[0][1] = 2
        self.positions = [
            [0, 0],  # p1
            [0, 1]  # p2
        ]
        self.agents = self.possible_agents = [f"agent_{i + 1}" for i in range(num_agents)]

        self.obs_spaces = gym.spaces.Box(
            low=0,
            high=8,
            shape=(self.height, self.width),
            dtype=np.uint8
        )
        self.observation_spaces = {
            # up down left right
            "agent_1": self.obs_spaces,
            "agent_2": self.obs_spaces,
        }

        self.action_spaces = {
            # up down left right
            "agent_1": gym.spaces.Discrete(4),
            "agent_2": gym.spaces.Discrete(4),
        }

        #use config from outside
        # if config.get("sheldon_cooper_mode"):
        #     #do something

    def reset(self, *, seed=None, options=None):
        self.chip =  np.zeros((10, 10), dtype=np.uint8)
        self.chip[0][0] = 1
        self.chip[0][1] = 2
        self.positions = [
            [0,0],#p1
            [0,1]#p2
        ]
        obs = {
            "agent_1": self.chip,
            "agent_2": self.chip
        }

        return obs, {}


    # __sphinx_doc_5_begin__
    def step(self, action_dict):

        move1 = action_dict["agent_1"]
        move2 = action_dict["agent_2"]
        self.move(1,move1)
        self.move(2,move2)


        obs = {
            "agent_1": self.chip,
            "agent_2": self.chip,
        }
        rewards = self.reward_function()

        terminateds = {"__all__": False}
        truncated = {}
        infos = {}

        return obs,rewards,terminateds,truncated,infos

    def reward(self):
        pass

    def move(self, player: int, act:int):
        old_pos = self.positions[player-1]
        if act ==0:
            new_pos = [old_pos[0], old_pos[1]+1]
        elif act ==1:
            new_pos = [old_pos[0], old_pos[1]-1]
        elif act ==2:
            new_pos = [old_pos[0]-1, old_pos[1]]
        elif act ==3:
            new_pos = [old_pos[0]+1, old_pos[1]]

        #if new_post out of matrix
        if (new_pos[0] < 0 or new_pos[0] >= 10 or new_pos[1] < 0 or new_pos[1] >= 10 or
                self.chip[new_pos[0]][new_pos[1]] != 0):
            return False
        else:
            try:
                self.positions[player-1] = new_pos
                self.chip[old_pos[0]][old_pos[1]] = 0
                self.chip[new_pos[0]][new_pos[1]] = player
            except Exception as e:
                pprint(f"Error: {e}")
                return False

            return True

    def reward_function(self):
        #todo call rfx

        reward = (self.positions[0][0]-self.positions[1][0])**2 + (self.positions[0][1]-self.positions[1][1])**2
        reward = np.sqrt(reward)
        rewards = {
            "agent_1": reward,
            "agent_2": reward,
        }

        return rewards

if __name__ == '__main__':
    env = Env_0()
    env.reset()
    action1 = env.action_spaces['agent_1'].sample()
    action2 = env.action_spaces['agent_2'].sample()
    act = {
        "agent_1": action1,
        "agent_2": action2
    }
    obs, rewards, terminateds, truncated, infos = env.step(act)
    print(f"obs: {obs}")
    print(f"rewards: {rewards}")
    print(f"terminateds: {terminateds}")
    print(f"truncated: {truncated}")
    print(f"infos: {infos}")
