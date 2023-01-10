_base_ = 'resnet/resnet18_8xb32_in1k.py'

dataset_type = 'CustomDataset'
classes = ['unripe', 'partially ripe', 'fully ripe']  # 数据集中各类别的名称

data = dict(
    train=dict(
        type=dataset_type,
        data_prefix="/home/hank/Desktop/Dataset/cropped_images",
        ann_file="/home/hank/Desktop/Dataset/cls_train.txt",
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        data_prefix="/home/hank/Desktop/Dataset/cropped_images",
        ann_file='/home/hank/Desktop/Dataset/cls_val.txt',
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        data_prefix="/home/hank/Desktop/Dataset/cropped_images",
        ann_file='/home/hank/Desktop/Dataset/cls_val.txt',
        classes=classes,
    )
)