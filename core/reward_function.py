import math

import numpy as np
import queue

from sympy.abc import alpha

from collections import deque





class RewardFunction:
    def __init__(self):
        pass
    '''
    引导 agent 逐步逼近目标位置来计算reward
    '''

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

    def _recent_dist_avg(self):
        return np.mean(self.recent_dist)

    def rfv2(self,init_dist,last_dist,avg_dist,dist):
        #update recent dist
        k1 = (dist - init_dist) / avg_dist
        k2 = (dist - last_dist) / avg_dist

        # if abs(k1)>=0.5:
        #     k1 = 0.5 if k1 > 0 else -0.5
        #
        # if abs(k2)>=0.5:
        #     k2 = 0.5 if k2 > 0 else -0.5
        k1 = np.log(abs(k1+1.0001)) if k1 > 0 else -np.log(abs(k1-1.0001))
        k2 = np.log(abs(k2+1.0001)) if k2 > 0 else -np.log(abs(k2-1.0001))


        if k2 > 0:
            r = (math.pow((1 + k2), 2) - 1) * (1 + np.tanh(k1))
        elif k2 < 0:
            '''
            防止 agent 通过distance的波动获取 reward
            '''
            r = -1 * (math.pow((1 - k2), 2) - 1) * (1 - np.tanh(k1)) - 0.05
            if dist - last_dist <= 1:
                r *= 1.25
        else:
            r = -0.05

        return r



def test_rf(distance: list):
    total = 0
    rf = RewardFunction()
    init_dist = distance[0]
    last_dist = init_dist
    for i in range(1,len(distance)):
        dist = distance[i]
        reward = rf.rfv2(init_dist=init_dist,last_dist=last_dist, dist=dist)
        last_dist = dist

        total *= 0.99
        total += reward
        print("reward: ", reward,'total:',total)


if __name__ == '__main__':

    #dist=[1,2,3,4,5]

    #dist=[1,2,1,2,1,2,1,2]
    dist = [4,10,5,10,5,4, 8,11]
    #dist = [5,5,4,4,5,5,4]
    test_rf(dist)

