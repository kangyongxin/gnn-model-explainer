from __future__ import print_function
import torch
#基本输入输出
# x = torch.tensor([5.5, 3])
# x = x.new_ones(5, 3, dtype=torch.double)  
# print("x",x)
# y = torch.rand(5, 3)
# z=x+y
# print("z",z)

# print("torch x+ y",torch.add(x,y))
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print("result",result)
# #任何使张量会发生变化的操作都有一个前缀 ‘_’。
# # #例如：x.copy_(y), x.t_(), 将会改变 x.
# print("result_t",result.t_())
# print("result",result)
#autograd 
# x = torch.ones(2, 2, requires_grad=True)
# print("x",x)

# y = x + 2
# print("y",y)

# z = y * y * 3
# out = z.mean()

# print("z",z)

# print("out", out)

# out.backward(torch.tensor(1.))

# print("x.grad",x.grad)
# print("y.grad",y.grad)#why?


x = torch.randn(3, requires_grad=True)
print("x",x)
y = x * 2
print("y.norm()",y.data.norm())
while y.data.norm() < 1000:
    y = y * 2

print("y",y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
