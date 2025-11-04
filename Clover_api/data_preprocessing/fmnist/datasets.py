import logging
import numpy as np
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
from PIL import Image

class FashionMNIST_truncated(data.Dataset):
    def __init__(self, root, cache_data_set=None, dataidxs=None, 
                train=True, transform=None, target_transform=None, download=False):
        
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__(cache_data_set)

    def __build_truncated_dataset__(self, cache_data_set):
        # 加载FashionMNIST数据集
        if cache_data_set is None:
            fashionmnist_dataobj = FashionMNIST(self.root, self.train, self.transform, 
                                              self.target_transform, self.download)
        else:
            fashionmnist_dataobj = cache_data_set

        # 转换为numpy数组并添加通道维度 (FashionMNIST原始格式为28x28)
        data = fashionmnist_dataobj.data.numpy()[:, :, :, None]  # 形状变为(N,28,28,1)
        target = np.array(fashionmnist_dataobj.targets)

        # 数据索引选择
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        
        # 转换为PIL图像对象 (单通道L模式)
        # img = Image.fromarray(img.squeeze().astype(np.uint8), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)