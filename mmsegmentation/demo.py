import os
import time
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import cv2

config_file = 'configs/unet.py'
checkpoint_file = 'work_dirs/unet/latest.pth'

# 通过配置文件和模型权重文件构建模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

root_dir = "/home/hank/Desktop/Dataset/seg/val/images"
all_time = 0
index = 0
for filename1 in os.listdir(root_dir):
    img = os.path.join(root_dir, filename1)
    # 对单张图片进行推理并展示结果
    # img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    start_time = time.time()
    result = inference_segmentor(model, img)
    index += 1
    all_time += (time.time() - start_time)
    # 在新窗口中可视化推理结果
    image = model.show_result(img, result, show=False)
    # cv2.imshow("image", image)
    # cv2.waitKey()
print(all_time/index)


