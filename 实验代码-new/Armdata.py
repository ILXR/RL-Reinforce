#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Yixuehao
"""

import random
# import csv
import pandas as pd
import numpy as np


# class BernoulliArm(object):
#
#     def __init__(self, p):
#         self.p = p
#
#     def step(self):
#
#         BernoulliArmindex = random.random()
#         print('BernoulliArmindex:', BernoulliArmindex)
#
#         if BernoulliArmindex < self.p:
#             return 1.0
#         else:
#             return 0.0


# 关于深度模型的切割实验

class DNNPartition():

    # 导入数据集

    def __init__(self, arm_k):

        self.arm_k =arm_k
        self.p = 0
        self.dnn_reader = pd.read_csv(open("./实验代码-new/DNN_Partition.csv"))
        self.dnn_data = np.array(self.dnn_reader)

   # 此处是获取奖励值
    def step(self, p):

        temp = random.randrange(7)

        print(p+temp*23)


        return self.dnn_data[p+temp*23, 6]