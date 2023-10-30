import sys

import tvm
import tvm.testing
from tvm import te
import numpy as np

print(tvm.cuda().exist)
run_cuda = True
if run_cuda:
    # Change this target to the correct backend for you gpu. For example: cuda (NVIDIA GPUs),
    # rocm (Radeon GPUS), OpenCL (opencl).
    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")

    # Recreate the schedule
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    print(type(C))
    s = te.create_schedule(C.op)
    bx, tx = s[C].split(C.op.axis[0], factor=64)


    # Finally we must bind the iteration axis bx and tx to threads in the GPU  compute grid.
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

    fadd = tvm.build(s, [A, B, C], target=tgt_gpu, name="myadd")

    ######################################################################
    # The compiled TVM function exposes a concise C API that can be invoked from any language.
    # We provide a minimal array API in python to aid quick testing and prototyping.
    # The array API is based on the `DLPack <https://github.com/dmlc/dlpack>`_ standard.
    ######################################################################
    dev = tvm.device(tgt_gpu.kind.name, 0) # - We first create a GPU device.
    n = 32768
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)    # tvm.nd.array copies the data to the GPU.
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd(a, b, c)    # - ``fadd`` runs the actual computation
    evaluator = fadd.time_evaluator(fadd.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%f" % (mean_time))
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
    # - ``numpy()`` copies the GPU array back to the CPU (so we can verify correctness).
    # Note that copying the data to and from the memory on the GPU is a required step.


    # You can inspect the generated code in TVM. The result of tvm.build is a TVM
    # Module. fadd is the host module that contains the host wrapper, it also
    # contains a device module for the CUDA (GPU) function.
    #
    # The following code fetches the device module and prints the content code.

    if (
        tgt_gpu.kind.name == "cuda"
        or tgt_gpu.kind.name == "rocm"
        or tgt_gpu.kind.name.startswith("opencl")
    ):
        dev_module = fadd.imported_modules[0]
        print("-----GPU code-----")
        # print(dev_module.get_source())  #cuda代码
    else:
        print(fadd.get_source())

# ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
#Saving and Loading Compiled Modules

# from tvm.contrib import cc
# from tvm.contrib import utils
#
# temp = utils.tempdir()
# fadd.save(temp.relpath("myadd.o"))
# if tgt_gpu.kind.name == "cuda":
#     fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
# if tgt_gpu.kind.name == "rocm":
#     fadd.imported_modules[0].save(temp.relpath("myadd.hsaco"))
# if tgt_gpu.kind.name.startswith("opencl"):
#     fadd.imported_modules[0].save(temp.relpath("myadd.cl"))
#
# cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])  # warning LNK4272:库计算机类型“x86”与目标计算机类型“x64”冲突
# print(temp.listdir())

# ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？

# # Load Compiled Module
# fadd1 = tvm.runtime.load_module(temp.relpath("myadd.so"))
# if tgt_gpu.kind.name == "cuda":
#     fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.ptx"))
#     fadd1.import_module(fadd1_dev)
#
# if tgt_gpu.kind.name == "rocm":
#     fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.hsaco"))
#     fadd1.import_module(fadd1_dev)
#
# if tgt_gpu.kind.name.startswith("opencl"):
#     fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.cl"))
#     fadd1.import_module(fadd1_dev)
#
# fadd1(a, b, c)
# tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
#
#
# # Pack Everything into One Library
#   # export everything as one shared library. pack the device modules into binary blobs
#   # and link them together with the host code. Currently we support packing of Metal, OpenCL and CUDA modules.
# fadd.export_library(temp.relpath("myadd_pack.so"))
# fadd2 = tvm.runtime.load_module(temp.relpath("myadd_pack.so"))
# fadd2(a, b, c)
# tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

# ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？


