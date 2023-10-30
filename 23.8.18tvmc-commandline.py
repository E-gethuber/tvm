#---------------------------tvm英文文档学习------------------------------------------
# 使用tvmc的例子   linux版
# https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html
# python -m tvm.driver.tvmc <option> ==  tvmc  <option>
# 加载onnx模型转为 relay ，之后编译到底层，运行，自动微调等等
# ---------------------------------------------------------------------------------

import tvm.driver.tvmc

# -------------------------preprocess↓--------------------------------------------
# from tvm.contrib.download import download_testdata
# from PIL import Image
# import numpy as np
#
# img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
# img_path = download_testdata(img_url, "imagenet_cat.png", module="data")
#
# # Resize it to 224x224
# resized_image = Image.open(img_path).resize((224, 224))
# img_data = np.asarray(resized_image).astype("float32")
#
# # ONNX expects NCHW input, so convert the array
# img_data = np.transpose(img_data, (2, 0, 1))
#
# # Normalize according to ImageNet
# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_stddev = np.array([0.229, 0.224, 0.225])
# norm_img_data = np.zeros(img_data.shape).astype("float32")
# for i in range(img_data.shape[0]):
#       norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]
#
# # Add batch dimension
# img_data = np.expand_dims(norm_img_data, axis=0)
#
# # Save to .npz (outputs imagenet_cat.npz)
# np.savez("imagenet_cat", data=img_data)
# -------------------------preprocess↑--------------------------------------------


# -------------------------postprocess↓--------------------------------------------
import os.path
import numpy as np

from scipy.special import softmax

from tvm.contrib.download import download_testdata

# Download a list of labels
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

output_file = "predictions.npz"

# Open the output and read the output tensor
if os.path.exists(output_file):
    with np.load(output_file) as data:
        scores = softmax(data["output_0"])
        scores = np.squeeze(scores)
        ranks = np.argsort(scores)[::-1]

        for rank in ranks[0:5]:
            print("class='%s' with probability=%f" % (labels[rank], scores[rank]))

# -------------------------postprocess↑--------------------------------------------
# import onnx
#
# onnx_model = onnx.load_model(r'models/resnet50/resnet50-v2-7.onnx')
# onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
# onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
# onnx.checker.check_model(onnx_model)
# onnx.save(onnx_model, r'models/resnet50/resnet50-v2-7-frozen.onnx')
# print('frozen model saved.')

