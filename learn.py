# 导入TVM和Pytorch并加载ResNet18模型
import torch
import torchvision
import tvm
import time
from tvm import relay
import numpy as np
from tvm.contrib.download import download_testdata

# Load a pretrained PyTorch model
# -------------------------------
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
# getattr 函数可以根据给定的对象和属性名称，返回该属性对应的值。这里传入的对象是 torchvision.models 模块返回预训练模型
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()


#载入测试图片
from PIL import Image

img_path = "img/img_2.png"
# img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #out[i] = (input[i]-mean[i])std[i]
    ]
)
img = my_preprocess(img)
# 新增Batch维度
img = np.expand_dims(img, 0)


# Relay导入TorchScript模型并编译到LLVM后端
######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph. The input name can be arbitrary.
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)


######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
target = "llvm"
target_host = "llvm"
ctx = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)


# 在目标硬件上进行推理并输出分类结果
######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.
import tvm.contrib.graph_executor as runtime

tvm_t0 = time.perf_counter() #time.clock()获取时间的方法在python3.x以后没有了
for i in range(10):
    dtype = "float32"
    m = runtime.GraphModule(lib["default"](ctx))
    # Set inputs
    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    # Execute
    m.run()
    # Get outputs
    tvm_output = m.get_output(0)
tvm_t1 = time.perf_counter()




# 1000类别分类
#####################################################################
# Look up synset name
# -------------------
# Look up prediction top 1 index in 1000 class synset.
# synset_url = "".join(
#     [
#         "https://raw.githubusercontent.com/Cadene/",
#         "pretrained-models.pytorch/master/data/",
#         "imagenet_synsets.txt",
#     ]
# )
# synset_name = "imagenet_synsets.txt"
# synset_path = download_testdata(synset_url, synset_name, module="data")
synset_path = 'class/imagenet_synsets.txt'
with open(synset_path) as f:
    synsets = f.readlines()  #展开成一行，每行以\n结尾：
#'n02119789 kit fox, Vulpes macrotis\n', 'n02100735 English setter\n',

synsets = [x.strip() for x in synsets]#strip用于删除字符串首尾指定字符（默认为空格）
#'n02119789 kit fox, Vulpes macrotis', 'n02100735 English setter'

splits = [line.split(" ") for line in synsets]#空格分开每一行，每个部分是一个字符串，每一行是一个list
#['n02119789', 'kit', 'fox,', 'Vulpes', 'macrotis']

key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

# class_url = "".join(
#     [
#         "https://raw.githubusercontent.com/Cadene/",
#         "pretrained-models.pytorch/master/data/",
#         "imagenet_classes.txt",
#     ]
# )
# class_name = "imagenet_classes.txt"
# class_path = download_testdata(class_url, class_name, module="data")
class_path = 'class/imagenet_classes.txt'
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Get top-1 result for TVM
top1_tvm = np.argmax(tvm_output.asnumpy()[0])
print(top1_tvm)
tvm_class_key = class_id_to_key[top1_tvm]
print(tvm_class_key)


# 在pytorch上跑一次，对比结果
# Convert input to PyTorch variable and get PyTorch result for comparison
torch_t0 = time.perf_counter()
for i in range(10):
    with torch.no_grad():
        torch_img = torch.from_numpy(img)
        output = model(torch_img)

        # Get top-1 result for PyTorch
        top1_torch = np.argmax(output.numpy())
        torch_class_key = class_id_to_key[top1_torch]
torch_t1 = time.perf_counter()

tvm_time = tvm_t1 - tvm_t0
torch_time = torch_t1 - torch_t0

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))
print('Relay time: ', tvm_time / 10.0, 'seconds')
print('Torch time: ', torch_time / 10.0, 'seconds')