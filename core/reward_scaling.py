import numpy as np


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n )

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


def test_reward_scaling_with_reset():
    rs = RewardScaling(shape=1, gamma=0.9)

    # Episode 1
    rewards1 = [-0.05,0.2,0.2,0.2]
    for r in rewards1:
        r = rs(r)
        print(r)

    # Episode 2
    rs.reset()
    rewards2 = [4, 5, 6]
    scaled_rewards = []
    for r in rewards2:
        scaled = rs(r)
        scaled_rewards.append(scaled)

    # Check that scaling is working in second episode
    assert len(scaled_rewards) == len(rewards2)
    assert not np.isnan(scaled_rewards[-1])
if __name__ == '__main__':
    #test_reward_scaling_with_reset()
    rs = RewardScaling(shape=1, gamma=0.9)
    for i in range(5):
        print(rs(i))