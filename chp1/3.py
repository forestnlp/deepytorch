#!/usr/bin/env python
# _*_coding:utf-8_*_

'''
@Time :    2021/1/28 9:48
@Author:  user
'''

import torch
import torch.tensor as ts

# a scalar

a = ts(42.)
print(a.dim(),a.item(),2*a)

# v vector

v = ts([1,2,3,4,3,2,1])
print(v.dim(),v.size(),v)

# m matrix

m = ts([[1,2,3],[2,3,4],[3,4,5],[5,6,7]])
print(m.dim(),m.size(),m)
