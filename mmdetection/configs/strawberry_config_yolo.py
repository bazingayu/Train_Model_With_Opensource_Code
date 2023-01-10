_base_ = 'yolo/yolov3_d53_320_273e_coco.py'

model = dict(
    bbox_head=dict(num_classes=1),
)
dataset_type='COCODataset'
classes = ('strawberry', )
data = dict(
    train=dict(
        img_prefix="/home/hank/Desktop/Dataset/images/",
        classes = classes,
        ann_file = "/home/hank/Desktop/Lectures/CV/Project/strawberry_train.json",
    ),
    val = dict(
        img_prefix="/home/hank/Desktop/Dataset/images/",
        classes = classes,
        ann_file = "/home/hank/Desktop/Lectures/CV/Project/strawberry_val.json",
    ),
    test = dict(
        img_prefix="/home/hank/Desktop/Dataset/images/",
        classes = classes,
        ann_file = "/home/hank/Desktop/Lectures/CV/Project/strawberry_val.json",
    ),
)