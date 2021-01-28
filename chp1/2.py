#!/usr/bin/env python
# _*_coding:utf-8_*_

'''
@Time :    2021/1/28 9:11
@Author:  user
'''
import torch
import numpy as np

x = [i for i in range(11)]

x_train = np.array(x, dtype=np.float32)

x_train = x_train.reshape(-1, 1)

y = [3 * i + 5 for i in x]

y_train = np.array(y, dtype=np.float32)

y_train = y_train.reshape(-1, 1)

print(x_train.shape, y_train.shape)

import torch.nn as nn

class LRModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LRModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out



intput_dim = 1
output_dim = 1

model = LRModel(intput_dim, output_dim)

# print(model)

epochs = 1000

learning_rate = 0.01

optim_sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

for epoch in range(epochs):
    epoch+=1

    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    optim_sgd.zero_grad()

    outputs = model(inputs)
    #outputs = model.forward(inputs)  #好像这2中方法都可以呀

    loss = criterion(outputs,labels)

    loss.backward()

    optim_sgd.step()

    if(epoch%50==0):
        print('epoch{},loss{}'.format(epoch,loss.item()))


for param in model.parameters():
    print(param)

torch.save(model.state_dict(),'model.pkl')

model2 = LRModel(intput_dim,output_dim)
model2.load_state_dict(torch.load('model.pkl'))
y_pred = model2(torch.from_numpy(x_train))
print(y_pred)