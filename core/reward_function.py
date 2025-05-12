import math

import numpy as np


class RewardFunction:

    #default
    @staticmethod
    def rfv1(init_dist,last_dist,dist):
        k1 = (init_dist - dist) / init_dist
        k2 = (last_dist - dist) / (dist+last_dist)

        if k2 > 0:
            r = (math.pow((1 + k2), 2) - 1) * (1 + np.tanh(k1))
        elif k2 < 0:
            '''
            防止 agent 通过distance的波动获取 reward
            '''
            r =  -1*(math.pow((1 - k2), 2) - 1) * (1 - np.tanh(k1)) - 0.05
            if dist - last_dist <= 1:
                r *= 1.25
        else:
            r = -0.05

        return r

def test_rf(distance: list):
    init_dist = distance[0]
    gamma = 0.99
    total = 0
    for i in range(1,len(distance)):
        last_dist = distance[i-1]
        dist = distance[i]
        reward = RewardFunction.rfv1(init_dist,last_dist,dist)
        total *= gamma
        total += reward

        print("reward: ", reward,'total:',total)




if __name__ == '__main__':
    dist=[8,8,16,7,100,6]

    test_rf(dist)