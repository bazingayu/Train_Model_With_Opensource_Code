from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyDataset(CustomDataset):
    # 写你实际的类别名就好了，跟生成mask是映射的数字顺序一致即可，有背景不需要改没有背景记得与生成mask时一样一定要在第一个加上background
    CLASSES = (
       'bg', 'fg'
    )
    # 这个数量与上面个数对应就好了,只是最后的预测每个类别对应的mask颜色
    PALETTE = [[0, 0 , 0], [1, 1, 1]]

    def __init__(self, **kwargs):
        super(MyDataset, self).__init__(
            **kwargs
        )