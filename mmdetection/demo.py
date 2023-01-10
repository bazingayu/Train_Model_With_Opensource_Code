import os
import time
from mmdet.apis import init_detector, inference_detector
import cv2

# model_name = "faster_rcnn"
model_name = "faster_rcnn"
# model_name = "yolo"
root_dir = "/home/hank/Desktop/Dataset/Group9/images"
if model_name == "faster_rcnn":
    config_file = 'configs/strawberry_config.py'
    checkpoint_file = 'work_dirs/strawberry_config/epoch_12.pth'
elif model_name == "ssd":
    config_file = 'configs/strawberry_config_ssd.py'
    checkpoint_file = 'work_dirs/strawberry_config_ssd/epoch_24.pth'
else:
    config_file = 'work_dirs/strawberry_config_yolo/strawberry_config_yolo.py'
    checkpoint_file = 'work_dirs/strawberry_config_yolo/latest.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
total_time = 0
index = 0
for filename1 in os.listdir(root_dir):
    filename = os.path.join(root_dir, filename1)
    image = cv2.imread(filename)
    image = cv2.resize(image, (512, 512))
    start_time = time.time()
    result = inference_detector(model, image)
    total_time += (time.time()-start_time)
    index += 1
    # res_image = model.show_result(filename, result)
    # cv2.imshow("win", res_image)
    # cv2.waitKey()

print(total_time/index)