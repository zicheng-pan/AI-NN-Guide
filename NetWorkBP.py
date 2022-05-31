import numpy as np


class Network(object):
    # size【】 的定义是：数组的个数代表整个神经网络有多少层，里面的数值代表每一层里面有多少个神经元（多少个参数，维度）
    def __init__(self, sizes):
        # 取列表长度，就是神经网络的层数
        self.num_layers = len(sizes)
        # 每层神经元的个数
        self.sizes = sizes
        # 初始化每层的偏置， 就是wx + b 中的b 的值,
        # [
        #   隐藏层的神经元的个数对应多少个偏置，初始化神经元没有偏置值，并且每个神经元对应一个偏置值
        #    所以偏置只有1列，这里初始化数据使用np.random.randn(y,1)
        #   后面每一个计算wx+b每一行w都对应一个b，所以需要（网络层数-1）这么多列的数据,每一列都有对应神经元个数的b的矩阵
        # ]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        #   上面一行相当于下面三行，用来初始化b的值得
        # self.biases = []
        # for y in self.size[1:]:
        #     self.biases.append(np.random.randn(y,1)) 使用标准正泰分布来初始化 y*1的数组

        #   初始化每一层的权重
        #   【3 2 1】这样的神经元，那么输入3个神经元，输出2个神经元 那么需要 3 * 2 的矩阵作为连接
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]

        pass


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# 创建神经网络的类

net = Network([3,2,1])
print(net.num_layers)
print(net.sizes)
print(net.biases)
print(net.weights)