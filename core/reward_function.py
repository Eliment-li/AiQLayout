import math

import numpy as np
import queue

from sympy.abc import alpha

from collections import deque

from utils.calc_util import SlideWindow


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
        if avg_dist == 0:
            avg_dist = init_dist
        #update recent dist
        k1 = (dist - init_dist) / avg_dist
        k2 = (dist - last_dist) / avg_dist

        k1 = np.log(abs(k1+1.0001)) if k1 > 0 else -np.log(abs(k1-1.0001))
        k2 = np.log(abs(k2+1.0001)) if k2 > 0 else -np.log(abs(k2-1.0001))


        if k2 > 0:
            r = (math.pow((1 + k2), 2) - 1) * (1 + np.tanh(k1))
        elif k2 < 0:
            '''
            防止 agent 通过distance的波动获取 reward
            '''
            r = -2 * (math.pow((1 - k2), 2) - 1) * (1 - np.tanh(k1)) - 0.05
        else:
            r = -0.05

        return r



def test_rf(distance: list):
    sw = SlideWindow(5)
    total = 0
    rf = RewardFunction()

    last_dist = distance[0]
    for i in range(1,len(distance)):
        dist = distance[i]
        reward = rf.rfv2(init_dist=distance[0],last_dist=last_dist, dist=dist,avg_dist=sw.current_avg)
        last_dist = dist

        total *= 0.99
        total += reward

        sw.next(dist)
        print("reward: ", reward,'total:',total)


if __name__ == '__main__':
    dist = [5,6,5,6,5,6,5,10,5,10,5]
    test_rf(dist)

