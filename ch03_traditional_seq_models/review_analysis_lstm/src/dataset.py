import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import *

# 自定义数据集类
class ReviewAnalysisDataset(Dataset):
    # 初始化
    def __init__(self, file_path):
        self.data = pd.read_json(file_path, lines=True).to_dict(orient='records')

    # 获取长度
    def __len__(self):
        return len(self.data)

    # 获取元素
    def __getitem__(self, idx):
        input = torch.tensor(self.data[idx]['review'])
        target = torch.tensor(self.data[idx]['label'], dtype=torch.float)
        return input, target

# 获取DataLoader
def get_dataloader(train=True):
    # 根据参数判断获取哪个数据集
    file_path = PROCESSED_DATA_DIR / (TRAIN_DATA_FILE if train else TEST_DATA_FILE)
    # 创建数据集
    dataset = ReviewAnalysisDataset(file_path)
    # 创建加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


if __name__ == '__main__':
    # 测试数据集
    train_dataset = ReviewAnalysisDataset(PROCESSED_DATA_DIR/TRAIN_DATA_FILE)
    print(len(train_dataset))
    print(train_dataset[0])

    # 测试数据加载器
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    # 取一批数据
    for input, target in train_loader:
        print(input.shape, target.shape)
        break
    test_iter = iter(test_loader)
    input, target = next(test_iter)
    print(input.shape, target.shape)

