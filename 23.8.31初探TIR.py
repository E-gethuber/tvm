import IPython
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

#  TVMScript实现matmul+relu 的tir
# @tvm.script.ir_module 表示 MyModule 是一个 IRModule。IRModule 是在机器学习编译中保存张量函数集合的容器对象。
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        # 这里的 global_symbol 对应函数名，tir.noalias 是一个属性，表示所有的缓冲存储器不重叠。
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
            # 块 是 TensorIR 中的基本计算单位。值得注意的是，该块包含比普通 NumPy 代码更多的信息。
            # 一个块包含一组块轴（vi、vj、vk）和围绕它们定义的计算。
                vi = T.axis.spatial(128, i)# vi、vj 为空间轴
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)#vk 归约轴
            #[block_axis] = T.axis.[axis_type]([axis_range], [mapped_value])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

# print(type(MyModule))  #<class 'tvm.ir.module.IRModule'>
# type(MyModule["mm_relu"])   tvm.tir.function.PrimFunc
###############################################schedule优化###################################
# print(MyModule.script())
sch = tvm.tir.Schedule(MyModule)
block_Y = sch.get_block("Y", func_name="mm_relu")
i, j, k = sch.get_loops(block_Y)  #拿到block外面的循环
j0, j1 = sch.split(j, factors=[None, 4])  #将j轴分为 j0*j1两层层循环
# print(sch.mod.script())

sch.reorder(j0, k, j1)   #改变顺序成 j0 k j1 原来是 j0 j1 k,把j1放到最内层
# print(sch.mod.script())

block_C = sch.get_block("C", "mm_relu")
sch.reverse_compute_at(block_C, j0)   #将下面relu的块c  放到j0的内层
# print(sch.mod.script())

sch.decompose_reduction(block_Y, k)   #将k这个循环拆成单个for循环运行完之后再运行下一个循环
# print(sch.mod.script())


#################################################通过TE来实现并转为TIR
from tvm import te

A = te.placeholder((128, 128), "float32", name="A")
B = te.placeholder((128, 128), "float32", name="B")
k = te.reduce_axis((0, 128), "k")
Y = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
C = te.compute((128, 128), lambda i, j: te.max(Y[i, j], 0), name="C")


te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm_relu"})

MyModuleFromTE = tvm.IRModule({"mm_relu": te_func})

sch = tvm.tir.Schedule(MyModuleFromTE)
block_Y = sch.get_block("Y", func_name="mm_relu")
i, j, k = sch.get_loops(block_Y)  #拿到block外面的循环
j0, j1 = sch.split(j, factors=[None, 4])  #将j轴分为 j0*j1两层层循环
sch.reorder(j0, k, j1)   #改变顺序成 j0 k j1 原来是 j0 j1 k,把j1放到最内层
block_C = sch.get_block("C", "mm_relu")
sch.reverse_compute_at(block_C, j0)   #将下面relu的块c  放到j0的内层
# print(sch.mod.script())
sch.decompose_reduction(block_Y, k)   #将k这个循环拆成单个for循环运行完之后再运行下一个循环

print(MyModuleFromTE.script())
#########################################################################


# ----------------------构建与运行
dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
# a @ b is equivalent to np.matmul(a, b)
c_mm_relu = np.maximum(a_np @ b_np, 0)

# rt_lib = tvm.build(MyModule, target="llvm")  #runtime
rt_lib = tvm.build(MyModuleFromTE, target="llvm")  #runtime
a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((128, 128), dtype="float32")

func_mm_relu = rt_lib["mm_relu"]
func_mm_relu(a_nd, b_nd, c_nd)

np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

#----------------------------------------对比时间
rt_lib_after = tvm.build(sch.mod, target="llvm")
rt_lib_after["mm_relu"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

f_timer_before = rt_lib.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of MyModule %g sec" % f_timer_before(a_nd, b_nd, c_nd).mean)
f_timer_after = rt_lib_after.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed sch.mod %g sec" % f_timer_after(a_nd, b_nd, c_nd).mean)



