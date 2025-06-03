

class Agent():
    def __init__(self, agent_name: str,number:int):
        self.agent_name = agent_name

    def get_agent_name(self) -> str:
        return self.agent_name

    def set_agent_name(self, agent_name: str):
        self.agent_name = agent_name



class AgentsManager():
    def __init__(self, nums:int):
        self.agents = []
        for i in range(nums):
            agent_name = f"agent_{i+1}"
            self.agents[i] = Agent(agent_name, i+1)

    def __call__(self, agent_number):
        return self.agents[agent_number - 1]

    def get_agent_name(self) -> str:
        return self.agent_name

    def set_agent_name(self, agent_name: str):
        self.agent_name = agent_name