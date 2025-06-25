from copy import deepcopy

import numpy as np

from core.chip import Chip
from core.routing import a_star_path
from utils.calc_util import SlideWindow
from utils.circuit_util import get_gates


class Agent():
    def __init__(self, number:int,chip:Chip):
        self.number = number
        self.activate = False
        self.chip = chip
        self.sw = SlideWindow(50)
        self.done = False

    def reset(self):
        self.activate = False
        self.agent_total_r = 0
        self.max_total_r = -np.inf
        self.sw.reset()

        self.done = False
        # other_dist, self_dist,depth= self.compute_dist(self.number)
        # sum_dist = other_dist * 0.5 + self_dist * 0.5
        # self.min_dist = self_dist
        # self.init_dist = sum_dist

    def set_activate(self):
        assert not self.done , "trying to  activate an agent that is already done."
        self.activate = True




class AgentsManager():
    def __init__(self, num_agents:int,chip):
        self.chip = chip
        self.agents = []
        self.activate_agent = 1
        self.num_agents = num_agents
        for i in range(num_agents):
            self.agents.append(Agent(i+1,chip))

    def __call__(self, agent_number)->Agent:
        return self.agents[agent_number - 1]

    def act_agent(self):
        return self.agents[self.activate_agent - 1]

    def set_done(self, agent_number: int):
        self.agents[agent_number - 1].done = True

    def is_done(self, agent_number: int) -> bool:
        return self.agents[agent_number - 1].done

    def is_all_done(self) -> bool:
        return all(agent.done for agent in self.agents)

    def switch_next(self):
        self.activate_agent = ((self.activate_agent) % self.num_agents) + 1
        #check if all agent is done
        done = np.array([agent.done for agent in self.agents])
        assert not done.all(), "All agents are done, cannot switch to next agent."

        while self.is_done(self.activate_agent - 1):
            self.activate_agent = ((self.activate_agent) % self.num_agents) + 1

        self.switch_agent(self.activate_agent)
        return True

    def switch_agent(self, agent_number: int):
        if agent_number < 1 or agent_number > len(self.agents):
            raise ValueError(f"Agent number {agent_number} is out of range.")

        self.agents[self.activate_agent - 1].activate = False
        self.activate_agent = agent_number
        self.agents[self.activate_agent - 1].activate = True

    def reset_agents(self):
        for agent in self.agents:
            agent.reset()

        self.agents[0].set_activate()
        self.activate_agent = 1


if __name__ == '__main__':
    am = AgentsManager(10, Chip(5,5))
    for i in range(20):
        print(am.activate_agent)
        am.switch_next()