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
# y=torch.ones(2,2,requires_grad=True)
# y = x + 2
# print("y",y)

# z = y * y * 3
# out = z.mean()

# print("z",z)

# print("out", out)

# out.backward(torch.tensor(1.))

# print("x.grad",x.grad)
# print("y.grad",y.grad)#why?

#network constructor
# x = torch.randn(3, requires_grad=True)
# print("x",x)
# y = x * 2
# print("y.norm()",y.data.norm())
# while y.data.norm() < 1000:
#     y = y * 2

# print("y",y)

# v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
# y.backward(v)

# print(x.grad)

# 一个典型的神经网络训练过程包括以下几点：
# 1.定义一个包含可训练参数的神经网络
# 2.迭代整个输入
# 3.通过神经网络处理输入
# 4.计算损失(loss)
# 5.反向传播梯度到神经网络的参数
# 6.更新网络的参数，典型的用一个简单的更新方法：weight = weight – learning_rate *gradient

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # 先定义自己需要的网络模块
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)#这里只有一幅输入，应该可以batch 构建的
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)#为啥是16*5*5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def num_flat_features(self, x):
        #这是flatten 操作，没有模块化的定义吗？为啥是自己写的

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        #然后在forward 中把它拼起来，x的维度要与第一层相同
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#使用
net = Net()
print(net)
# Net(
#   (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#   (fc1): Linear(in_features=400, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )
params = list(net.parameters())
# for i in range(2):
#     print(params[i]) # conv1's .weight
print("params",params[1])
input = torch.randn(1, 1, 32, 32)



# net.zero_grad()
# out.backward(torch.randn(1, 10)) 

# output = net(input)


# net.zero_grad()     # zeroes the gradient buffers of all parameters

# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)

# loss.backward() #得到本轮梯度

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

# for i in range(2):
#     print(params[i]) # conv1's .weight

#用学习率对梯度进行加权之后与原先的权重详加，有专门的函数完成
#weight = weight - learning_rate * gradient

import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=1)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
print("output",output)
# loss = criterion(output, target)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
print("loss",loss)
loss.backward()
optimizer.step()    # Does the update

params1= list(net.parameters())
#print(params1[1])
print("params1",params1[1])
output1 = net(input)
print("output1",output1)
