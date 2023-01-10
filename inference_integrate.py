import os
from matplotlib import pyplot as plt
from mmdet.apis import init_detector, inference_detector
from mmseg.apis import inference_segmentor, init_segmentor
from mmcls.apis import inference_model, init_model, show_result_pyplot
import cv2

model_name = "faster_rcnn"
root_dir = "/home/hank/Desktop/Dataset/Group9/images"
# root_dir = "/home/hank/Desktop/Dataset/images"
if model_name == "faster_rcnn":
    config_file = 'mmdetection/configs/strawberry_config.py'
    checkpoint_file = 'mmdetection/work_dirs/strawberry_config/epoch_12.pth'
elif model_name == "ssd":
    config_file = 'mmdetection/configs/strawberry_config_ssd.py'
    checkpoint_file = 'mmdetection/work_dirs/strawberry_config_ssd/epoch_24.pth'
else:
    config_file = 'mmdetection/work_dirs/strawberry_config_yolo/strawberry_config_yolo.py'
    checkpoint_file = 'mmdetection/work_dirs/strawberry_config_yolo/latest.pth'

model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

segmentation_config_file = 'mmsegmentation/configs/unet.py'
segmentation_checkpoint_file = 'mmsegmentation/work_dirs/unet/latest.pth'
# 通过配置文件和模型权重文件构建模型
segmentation_model = init_segmentor(segmentation_config_file, segmentation_checkpoint_file, device='cuda:0')

cls_config = "mmclassification/work_dirs/resnet18_seg/resnet18_seg.py"
cls_checkpoint = "mmclassification/work_dirs/resnet18_seg/latest.pth"
# build the model from a config file and a checkpoint file
cls_model = init_model(cls_config, cls_checkpoint, "cpu")

fig = plt.figure()
for index, filename1 in enumerate(os.listdir(root_dir)):
    filename = os.path.join(root_dir, filename1)
    image = cv2.imread(filename)
    result = inference_detector(model, image)
    h, w, _ = image.shape
    count = 0
    for res in result[0]:
        # print(res)
        if(res[-1] > 0.7):
            width = res[3] - res[1]
            height = res[2] - res[0]
            crop_image = image[max(0, int(res[1] - height * 0.05)):min(int(res[3] + height * 0.05), h), max(0, int(res[0] - width * 0.05)):min(int(res[2] + width * 0.05), w)]

            result = inference_segmentor(segmentation_model, crop_image)

            # 在新窗口中可视化推理结果
            crop_image[result[0]==0] = 0
            count += 1

            # crop_image = cv2.resize(crop_image, (64, 64))
            result = inference_model(cls_model, crop_image)
            # show_result_pyplot(cls_model, crop_image, result)
            image = cv2.rectangle(image, (int(res[0]),int(res[1])), (int(res[2]) , int(res[3])), color=(0, 0, 255))
            image = cv2.putText(image, result['pred_class'], (int(res[2]), int(res[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    image = cv2.putText(image, "count : " + str(count), (0, image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 0), 4)
    cv2.imwrite(filename1, image)
    # plt.subplot(3, 2, index+1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()
# plt.savefig(f'{model_name}_result.png',bbox_inches='tight')
    # cv2.imshow("win", image)
    # cv2.waitKey()
    # res_image = model.show_result(filename, result)
    # cv2.imshow("win", res_image)
    # cv2.waitKey()