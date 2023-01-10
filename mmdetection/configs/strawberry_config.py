_base_ = 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
    )
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