# 自动求梯度
import torch
x = torch.rand(1,requires_grad=True)
w = torch.rand(1,requires_grad=True)
b = torch.rand(1)
y = x*w
z = y+b

print(x,x.is_leaf,x.requires_grad)
print(w,w.is_leaf,w.requires_grad)
print(y,y.is_leaf,y.requires_grad)
print(z,z.is_leaf,z.requires_grad)

#反向传播，传播z
z.backward()
print(w.grad,x.grad)