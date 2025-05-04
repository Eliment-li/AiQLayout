import math

import numpy as np


class RewardFunction:

    #default
    def rfv1(self,env,distance):
        k1 = (env.default_distance - distance) / env.default_distance
        k2 = (env.last_distance - distance) / env.last_distance

        if k2 > 0:
            r = (math.pow((1 + k2), 2) - 1) * (1 + np.tanh(k1))
        elif k2 < 0:
            '''
            防止 agent 通过distance的波动获取 reward
            '''
            r = -3.5 * (math.pow((1 - k2), 2) - 1) * (1 - np.tanh(k1))
            if distance - env.last_distance <= 1:
                r *= 1.25
        else:
            r = -0.05

        env.last_distance = distance
        return -r
