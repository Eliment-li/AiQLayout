import functools
from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding, EzPickle
from gymnasium.spaces import Discrete,Box
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers
from tianshou.env import PettingZooEnv


def new_env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = Env_1(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class Env_1(AECEnv):
    metadata = {"render_modes": ["human"], "name": "Env_1"}
    def __init__(self, render_mode=None, config=None):
        super().__init__()
        num_agents = 2
        # define chip
        self.height = 4
        self.width = 4
        self.channel = 1  # RGB 图像
        self.chip = None

        self.agents = self.possible_agents = [f"agent_{i + 1}" for i in range(num_agents)]
        # optional: a mapping between agent name and ID
        # mapping agent_0 to 0
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )



        self._obs_spaces = Box(
            low=0,
            high=8,
            shape=(self.height, self.width),
            dtype=np.uint8
        )
        self._observation_spaces = {
            # up down left right
            "agent_1": self._obs_spaces,
            "agent_2": self._obs_spaces,
        }

        self._action_spaces = {
            # up down left right
            "agent_1": Discrete(4),
            "agent_2": Discrete(4),
        }


        self.render_mode = render_mode
        #use config from outside
        # if config.get("sheldon_cooper_mode"):
        #     #do something
    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self._obs_spaces

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # We can seed the action space to make the environment deterministic.
        return Discrete(4)

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def reset(self, *, seed=None, options=None):
        """
               Reset needs to initialize the following attributes
               - agents
               - rewards
               - _cumulative_rewards
               - terminations
               - truncations
               - infos
               - agent_selection
               And must set up the environment so that render(), step(), and observe()
               can be called without issues.
               Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        ### 1
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        ###

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        ### custom
        self.chip =  np.zeros((self.height, self.width), dtype=np.uint8)
        self.chip[0][0] = 1
        self.chip[0][1] = 2
        self.positions = [
            [0,0],#p1
            [0,1]#p2
        ]
        # obs = {
        #     "agent_1": self.chip,
        #     "agent_2": self.chip
        # }
        ####

    def step(self, action):
        #### 1
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        ####
        agent = self.agent_selection
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            # self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
            #     (self.state[self.agents[0]], self.state[self.agents[1]])
            # ]
            for agent_i in self.agents:
                #暂时设置为0
                self.rewards[agent_i] = 0

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            for agent_i in self.agents:
                self.truncations[agent_i] = False

            terminations = {agent: False for agent in self.agents}

            self.num_moves += 1
            env_truncation = self.num_moves >= 100
            truncations = {agent: env_truncation for agent in self.agents}

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
        else:
            # necessary, so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = None
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
             self.render()

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
        if (new_pos[0] < 0 or new_pos[0] >= self.height or new_pos[1] < 0 or new_pos[1] >= self.width or
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
    def render(self) -> None | np.ndarray | str | list:
        pass
    def close(self):
        print("close env")

def test_env():
    env = new_env(render_mode="human")
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        print(
            f"agent: {agent}, observation: {observation}, reward: {reward}, termination: {termination}, truncation: {truncation}, info: {info}"
        )
        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
        env.step(action)
    env.close()

if __name__ == '__main__':
    # from pettingzoo.classic import rps_v2
    # from pettingzoo.classic import tictactoe_v3
    # from tianshou.env import PettingZooEnv
    #
    #
    env1 = PettingZooEnv(Env_1())
    # env = PettingZooEnv(tictactoe_v3.env(render_mode="human"))
    obs = env1.reset()

