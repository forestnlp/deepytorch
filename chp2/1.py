#!/usr/bin/env python
# _*_coding:utf-8_*_

'''
@Time :    2021/1/28 16:38
@Author:  user
'''

import torch


class ClassifyModel(torch.nn):
    def __init__(self, dim_x, dim_y):
        self.dimx = dim_x
        self.dimy = dim_y
        pass

    def forward(self, x):
        return x


dim_x = 8
dim_y = 1
model = ClassifyModel(dim_x, dim_y)
