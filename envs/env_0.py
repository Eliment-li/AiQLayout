
import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiDiscrete

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Env_0(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()

        #define chip
        num_players = 2
        self.agents = self.possible_agents = [f"player{i + 1}" for i in range(num_players)]


        # all positons of player

        self.obs = MultiDiscrete(np.array([2] * (num_players),dtype=int))
        self.observation_spaces = {
            "player1": self.obs,
            "player2": self.obs,
        }
        self.action_spaces = {
            # up down left right
            "player1": gym.spaces.Discrete(4),
            "player2": gym.spaces.Discrete(4),
        }

        self.positions = [
            [0,0],#p1
            [0,1]#p2
        ]

        #use config from outside
        # if config.get("sheldon_cooper_mode"):
        #     #do something

    # __sphinx_doc_4_begin__
    def reset(self, *, seed=None, options=None):
        self.positions = [
            [0,0],#p1
            [0,1]#p2
        ]

        return {"agent_1": self.obs, "agent_2": self.obs}, {}


    # __sphinx_doc_5_begin__
    def step(self, action_dict):
        # self.num_moves += 1
        #
        # move1 = action_dict["player1"]
        # move2 = action_dict["player2"]
        #
        # # Set the next observations (simply use the other player's action).
        # # Note that because we are publishing both players in the observations dict,
        # # we expect both players to act in the next `step()` (simultaneous stepping).
        # observations = {"player1": move2, "player2": move1}
        #
        # # Compute rewards for each player based on the win-matrix.
        # r1, r2 = self.WIN_MATRIX[move1, move2]
        # rewards = {"player1": r1, "player2": r2}
        #
        # # Terminate the entire episode (for all agents) once 10 moves have been made.
        # terminateds = {"__all__": self.num_moves >= 10}
        #
        # # Leave truncateds and infos empty.
        # return observations, rewards, terminateds, {}, {}

        move1 = action_dict["player1"]
        move2 = action_dict["player2"]

        #update obs

        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        # return {"agent_1": [obs of agent_1]}, {...}, ...

        pass

    def reward(self):
        pass


# __sphinx_doc_5_end__
