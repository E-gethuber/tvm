#---------------------------tvm英文文档学习------------------------------------------
# 1.使用tvmc的例子   win版
# https://tvm.apache.org/docs/tutorial/autotvm_relay_x86.html
#加载onnx模型转为 relay ，之后编译到底层，运行，自动微调等等


# --------------------------------------tvmc win版↓-------------------------------------------
import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import timeit


model_url = (
    "https://github.com/onnx/models/raw/main/"
    "vision/classification/resnet/model/"
    "resnet50-v2-7.onnx"
)

# model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
# onnx_model = onnx.load(model_path)
onnx_model = onnx.load('./models/resnet50/resnet50-v2-7.onnx')

# Seed numpy's RNG to get consistent results
np.random.seed(0)

img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
# img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# Resize it to 224x224
# resized_image = Image.open(img_path).resize((224, 224))
resized_image = Image.open('img/kitten.jpg').resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# Our input image is in HWC layout while ONNX expects CHW input, so convert the array
img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to the ImageNet input specification
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
img_data = np.expand_dims(norm_img_data, axis=0)

target = "llvm"

# The input name may vary across model types. You can use a tool
# like Netron to check input names
input_name = "data"
shape_dict = {input_name: img_data.shape}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
print('read onnx done')

with open('23-8-19/relay-frontend-from_onnx-mod.txt', 'w+')as f:
    f.write(str(mod))

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)



dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))


dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()



#Collect Basic Performance Data
timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)


#postprocess
from scipy.special import softmax

# Download a list of labels
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

# Open the output and read the output tensor
scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))


#tune the model
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm import autotvm
number = 10 # specifies the number of different configurations that we will test,
repeat = 1 # specifies how many measurements we will take of each configuration
min_repeat_ms = 0  #specifies how long need to run configuration test
# since we're tuning on a CPU, can be set to 0   ，gpu is ncessary not 0
timeout = 10  # in seconds  places an upper limit on how long to run training code for each tested configuration.

# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)


#Create a simple structure for holding tuning options     XGBoost Grid algorithm
tuning_option = {
    "tuner": "xgb",
    "trials": 20,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "resnet-50-v2-autotuning.json",
}




# begin by extracting the tasks from the onnx model
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

    # choose tuner
    tuner = "xgb"

    # create tuner
    if tuner == "xgb":
        tuner_obj = XGBTuner(task, loss_type="reg")
    elif tuner == "xgb_knob":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="knob")
    elif tuner == "xgb_itervar":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="itervar")
    elif tuner == "xgb_curve":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="curve")
    elif tuner == "xgb_rank":
        tuner_obj = XGBTuner(task, loss_type="rank")
    elif tuner == "xgb_rank_knob":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="knob")
    elif tuner == "xgb_rank_itervar":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="itervar")
    elif tuner == "xgb_rank_curve":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="curve")
    elif tuner == "xgb_rank_binary":
        tuner_obj = XGBTuner(task, loss_type="rank-binary")
    elif tuner == "xgb_rank_binary_knob":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="knob")
    elif tuner == "xgb_rank_binary_itervar":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="itervar")
    elif tuner == "xgb_rank_binary_curve":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="curve")
    elif tuner == "ga":
        tuner_obj = GATuner(task, pop_size=50)
    elif tuner == "random":
        tuner_obj = RandomTuner(task)
    elif tuner == "gridsearch":
        tuner_obj = GridSearchTuner(task)
    else:
        raise ValueError("Invalid tuner: " + tuner)

    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )


# Compiling an Optimized Model with Tuning Data
with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))


#Verify that the optimized model runs and produces the same results:
dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))


#Comparing the Tuned and Untuned Models
import timeit

timing_number = 10
timing_repeat = 10
optimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}


print("optimized: %s" % (optimized))
print("unoptimized: %s" % (unoptimized))

