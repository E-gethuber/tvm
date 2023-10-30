# #-----------------------------------------------矩阵加法-----------------------------------------------------------------
# import numpy as np
# import tvm
# from tvm.ir.module import IRModule
# from tvm.script import tir as T
# type = 'float32'
# # init data
# a = np.arange(16).reshape(4, 4).astype(type)
# b = np.arange(16, 0, -1).reshape(4, 4).astype(type)
# c = np.empty((4, 4), dtype="float32")
# c = a +b
# print(c)
#
# @tvm.script.ir_module
# class MyModule:
#     @T.prim_func
#     def m_add(A: T.Buffer((4, 4), "float32"),
#                 B: T.Buffer((4, 4), "float32"),
#                 C: T.Buffer((4, 4), "float32")):
#         T.func_attr({"global_symbol": "m_add", "tir.noalias": True})
#         for i, j in T.grid(4, 4):
#             with T.block("C"):
#                 vi = T.axis.spatial(4, i)
#                 vj = T.axis.spatial(4, j)
#                 C[vi, vj] += A[vi, vj] + B[vi, vj]
#
#
# a_nd = tvm.nd.array(a)
# b_nd = tvm.nd.array(b)
# c_nd = tvm.nd.empty((4, 4),dtype = type)
#
# rt_mod = tvm.build(MyModule, target='llvm')
# func_m_add = rt_mod['m_add']
# func_m_add(a_nd, b_nd, c_nd)
# np.testing.assert_allclose(c, c_nd.numpy(), rtol=1e-5)
# # -----------------------------------------------矩阵加法-----------------------------------------------------------------

# -----------------------------------------------conv-----------------------------------------------------------------
import numpy as np
import tvm
from tvm import relay
from tvm.ir.module import IRModule
from tvm.script import tir as T
#------------------------------------init
stride = 2
N, CI, H, W, CO, K = 2, 2, 8, 8, 2, 3
OUT_H, OUT_W = (H - K)//stride + 1, (W - K)//stride + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W).astype('float32')
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K).astype('float32')
out = np.zeros((N, CO, OUT_H, OUT_W),dtype="float32")
# print(data, '\n',weight)
#------------------------------------torch version
import torch
data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch, stride=2)
conv_torch = conv_torch.numpy().astype(np.int64)
# print(conv_torch)

#------------------------------------TIR version
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def myconv(myinput: T.Buffer((N, CI, H, W), 'float32'),
               myweight: T.Buffer((CO, CI, K, K), 'float32'),
               myoutput: T.Buffer((N, CO, OUT_H, OUT_W), 'float32')):
        T.func_attr({"global_symbol": "myconv", "tir.noalias": True})
        for i, j, l, m, k, n, o in T.grid(N, CO, OUT_H, OUT_W, CI,  K, K):
            with T.block("Y"):
                vi = T.axis.spatial(N, i)
                vj = T.axis.spatial(CO, j)
                vl = T.axis.reduce(OUT_H, l)
                vm = T.axis.spatial(OUT_W, m)
                vk = T.axis.spatial(CI, k)
                vn = T.axis.spatial(K, n)
                vo = T.axis.reduce(K, o)

                myoutput[vi, vj, vl, vm] = myoutput[vi, vj, vl, vm] + myinput[vi, vk, vl * stride + vn, vm * stride + vo] * myweight[vj, vk, vn, vo]

#-----------------------------------build&compare
a_nd = tvm.nd.array(data)
b_nd = tvm.nd.array(weight)
c_nd = tvm.nd.array(out)
rt_mod = tvm.build(MyModule, target="llvm")
prim_func = rt_mod['myconv']
prim_func(a_nd, b_nd, c_nd)
print(conv_torch)
print(c_nd.numpy())
np.testing.assert_allclose(conv_torch, c_nd.numpy(), rtol=1e-5)
#--------------------------------------------------------------------------------------------------------------------


