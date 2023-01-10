_base_ = [
    '_base_/models/fcn_unet_s5-d16.py',
    './_base_/default_runtime.py', './_base_/schedules/schedule_40k.py'
]
# dataset settings
dataset_type = 'MyDataset'
data_root = '/home/hank/Desktop/Dataset/seg' #填你生成的mask数据集的那个路径
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (64, 64)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(80, 80), keep_ratio=False), # img_scale根据自己的图片大小填写
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(64, 64),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(64, 64), keep_ratio=False),
            # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=400,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/images',
            ann_dir='train/mask',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/mask',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/mask',
        pipeline=test_pipeline)
)

model = dict(decode_head=dict(
        num_classes=2),
    auxiliary_head=dict(
        num_classes=2,))
evaluation = dict(metric='mDice')
