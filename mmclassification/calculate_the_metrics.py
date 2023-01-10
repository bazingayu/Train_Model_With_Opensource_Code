import mmcv
import numpy as np

from mmcls.datasets import build_dataset
from mmcls.core.evaluation import calculate_confusion_matrix
cfg = mmcv.Config.fromfile("work_dirs/resnet18_seg/resnet18_seg.py")
dataset = build_dataset(cfg.data.test)
pred = mmcv.load("./test.pkl")['class_scores']
print(pred)
matrix = calculate_confusion_matrix(pred, dataset.get_gt_labels())
print(matrix)
# tensor([[47.,  0.,  0.,  ...,  0.,  0.,  0.],
#         [ 0., 46.,  0.,  ...,  0.,  0.,  0.],
#         [ 0.,  0., 38.,  ...,  0.,  0.,  0.],
#         ...,
#         [ 0.,  0.,  0.,  ..., 36.,  0.,  0.],
#         [ 0.,  0.,  0.,  ...,  0., 24.,  0.],
#         [ 0.,  0.,  0.,  ...,  0.,  0., 26.]])
# >>> import matplotlib.pyplot as plt
# >>> plt.imshow(matrix[:20, :20])   # Visualize the first twenty classes.
# >>> plt.show()