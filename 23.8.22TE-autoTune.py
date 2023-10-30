import logging
import sys
import torch
import numpy as np
import tvm
from tvm import te
import tvm.testing

# the module is called `autotvm`
from tvm import autotvm
torch.nn.Linear
def matmul_basic(N, L, M, dtype):

    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis    #返回 除了规约轴之外的两个轴，即 return N，M
    k = s[C].op.reduce_axis[0]  #产生规约轴k，加不加[0]的结果一样

    yo, yi = s[C].split(y, 8)   #将y轴除以8，yi=(0,8) yo = (0,N,8)
    xo, xi = s[C].split(x, 8)
    s[C].reorder(yo, xo, k, yi, xi)  #交换循环顺序

    return s, [A, B, C]

matmul_basic(5, 10, 2, 'float32')
#这段程序是一个矩阵乘法的基本实现。它接受四个参数：N，L，M和dtype。其中，N表示矩阵A的行数，L表示矩阵A的列数和矩阵B的行数，M表示矩阵B的列数，dtype表示数据类型。
# 程序首先创建了两个占位符A和B，分别表示矩阵A和矩阵B。这些占位符用于表示计算图中的输入数据。
# 然后，程序定义了一个reduce_axis对象k，用于表示一个归约轴。该轴的范围是从0到L-1，用于对矩阵A和矩阵B进行归约计算。
# 接下来，程序定义了一个计算操作C，用于计算矩阵乘法的结果。计算操作C的形状是(N, M)，计算规则是对于每个元素(i, j)，计算A[i, k] * B[k, j]的累加和，其中k是归约轴。
# 然后，程序创建了一个调度对象s，并将计算操作C添加到调度对象中。
# 在调度部分，程序将计算操作C的轴进行了重新排序。首先，将y轴和x轴分割成更小的块yo和xo，每个块的大小为8。