# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
import time
import mmcv

from mmcls.apis import inference_model, init_model, show_result_pyplot



root_dir = "/home/hank/Desktop/Dataset/seg/val/images_seg/"

config = "work_dirs/resnet18_seg/resnet18_seg.py"
checkpoint = "work_dirs/resnet18_seg/latest.pth"
device = "cuda:0"


# build the model from a config file and a checkpoint file
model = init_model(config, checkpoint, device)
# test a single image
all_time = 0
index = 0
for filename1 in os.listdir(root_dir):
    filename = os.path.join(root_dir, filename1)
    start_time = time.time()
    result = inference_model(model, filename)
    all_time += (time.time() - start_time)
    index += 1
    # show_result_pyplot(model, filename, result)
print(all_time/index)

