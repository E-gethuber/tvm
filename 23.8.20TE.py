import tvm
import tvm.testing
from tvm import te
import numpy as np

tgt = tvm.target.Target(target="llvm", host="llvm")

n = te.var("n")                     #symbolic variable n to represent the shape
# n = tvm.runtime.convert(1024)     #The generated function will only take vectors with length 1024
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

# ↑no actual computation happens during this phase, as we are only declaring how the computation should be done

s = te.create_schedule(C.op)


fadd = tvm.build(s, [A, B, C], tgt, name="myadd")
# The build function takes the schedule, the desired
#signature of the function (including the inputs and outputs) as well as target language we want to compile to.


dev = tvm.device(tgt.kind.name, 0)   # - We first create a GPU device.
print('dev', dev)
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
# - Then tvm.nd.array copies the data to the device
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)

tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


##To get a comparison to numpy ↓
import timeit

np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "n = 32768\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(n, 1).astype(dtype)\n"
    "b = numpy.random.rand(n, 1).astype(dtype)\n",
    stmt="answer = a + b",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))

#
def evaluate_addition(func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    n = 32768
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))

    log.append((optimization, mean_time))
#
#
log = [("numpy", np_running_time / np_repeat)]
evaluate_addition(fadd, tgt, "naive", log=log)
##Numpy running time: 0.000005    naive: 0.000006


#  ↓Updating the Schedule to Use Parallelism
# print(tvm.lower(s, [A, B, C], simple_mode=True)) # generate the Intermediate Representation (IR) of the TE
factor = 4
outer, inner = s[C].split(C.op.axis[0], factor=factor)
# first we have to split the schedule into inner and outer loops using the split scheduling primitive
#The inner loops can use vectorization to use SIMD instructions using the vectorize scheduling primitive
# outer loops can be parallelized using the parallel scheduling primitive
s[C].parallel(outer)
s[C].vectorize(inner)
fadd_vector = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")
evaluate_addition(fadd_vector, tgt, "vector", log=log)


baseline = log[0][1]
print("%s\t%s\t%s" % ("Operator".rjust(20), "Timing".rjust(20), "Performance".rjust(20)))
for result in log:
    print(
        "%s\t%s\t%s"
        % (result[0].rjust(20), str(result[1]).rjust(20), str(result[1] / baseline).rjust(20))
    )