import math
from mimetypes import inited

import numpy as np
import queue

from sympy import pprint
from sympy.abc import alpha

from collections import deque

from core.reward_scaling import RewardScaling
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
            r = -0.01

        return r

    def _recent_dist_avg(self):
        return np.mean(self.recent_dist)

    #The Env_1 only compatible with  rf2 and vice versa
    def rfv2(self,init_dist,last_dist,avg_dist,dist):
        if avg_dist == 0:
            avg_dist = init_dist
        #update recent dist
        k1 = (dist - init_dist) / avg_dist
        k2 = (dist - last_dist) / avg_dist

        # print(f'k1: {k1}, k2: {k2}')
        k1 = np.log(abs(k1)+1) if k1 > 0 else -np.log(abs(k1)+1)
        k2 = np.log(abs(k2)+1) if k2 > 0 else -np.log(abs(k2)+1)
        # print(f'k1: {k1}, k2: {k2}')


        if k2 > 0:
            r = (math.pow((1 + k2), 2) - 1) * (1+abs(k1))
            # print('k2>0',(1 + abs(k1)))
        elif k2 < 0:
            r = (math.pow((1 - k2), 2) - 1) * (1 +abs(k1)) * -1.15 -0.1
            # print('k2<0',(1 + abs(k1)))
        else:
            r = -0.05

        return r


    def rfv3(self,init_dist,last_dist,avg_dist,dist):
        if avg_dist == 0:
            avg_dist = init_dist
        #update recent dist
        k1 = ( init_dist - dist) / avg_dist
        k2 = (last_dist - dist) / avg_dist

        k1 = np.log(abs(k1)+1) if k1 > 0 else -np.log(abs(k1)+1)
        k2 = np.log(abs(k2)+1) if k2 > 0 else -np.log(abs(k2)+1)

        if k2 > 0:
            r = (math.pow((1 + k2), 2) - 1) * (1 + abs(k1))
        elif k2 < 0:
            r = (math.pow((1 - k2), 2) - 1) * (1 +abs(k1)) * -1.15 -0.05
        else:
            r = -0.01
        return r

    def rfv4(self,init_dist,last_dist,avg_dist,dist):
        return -dist

    def rfv5(self,init_dist,last_dist,avg_dist,dist):
        if avg_dist == 0:
            avg_dist = init_dist
        #update recent dist
        k1 = ( init_dist - dist) / avg_dist
        k2 = (last_dist - dist) / avg_dist

        if k2 > 0:
            r = (math.pow((1 + k2), 2) - 1) * (1+abs(k1))
        elif k2 < 0:
            r = (math.pow((1 - k2), 2) - 1) * (1 +abs(k1)) * -1.05 -0.01
        else:
            r = -0.02
        return r




if __name__ == '__main__':
    init_dist = 483
    last_dist = init_dist
    dist = [556,616,553,566,551,550,532]
    sw = SlideWindow(50)
    rs = RewardScaling(shape=1, gamma=0.9)
    r_rec=[]
    rs_erc = []
    for d in dist:

        sw.next(d)
        r = RewardFunction().rfv3(init_dist=init_dist,last_dist=last_dist,dist=d,avg_dist=sw.current_avg)
        last_dist = d
        r_rec.append(r)
        rs_erc.append(rs(r))
    print(r_rec)
    print(rs_erc)