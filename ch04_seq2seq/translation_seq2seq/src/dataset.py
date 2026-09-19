import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import *

# 自定义数据集类
class TranslationDataset(Dataset):
    # 初始化
    def __init__(self, file_path):
        self.data = pd.read_json(file_path, lines=True).to_dict(orient='records')

    # 获取长度
    def __len__(self):
        return len(self.data)

    # 获取元素
    def __getitem__(self, idx):
        input = torch.tensor(self.data[idx]['zh'])
        target = torch.tensor(self.data[idx]['en'])
        return input, target

from torch.nn.utils.rnn import pad_sequence
# 自定义一个collate_fn方法
def collate_fn(batch):
    # 先拆分开input和target
    inputs = [ item[0] for item in batch ]
    targets = [item[1] for item in batch ]
    # 填充，并合并成一个Tensor
    input_tensor = pad_sequence(inputs, batch_first=True, padding_value=0)
    target_tensor = pad_sequence(targets, batch_first=True, padding_value=0)
    return input_tensor, target_tensor



# 获取DataLoader
def get_dataloader(train=True):
    # 根据参数判断获取哪个数据集
    file_path = PROCESSED_DATA_DIR / (TRAIN_DATA_FILE if train else TEST_DATA_FILE)
    # 创建数据集
    dataset = TranslationDataset(file_path)
    # 创建加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    # 测试数据集
    train_dataset = TranslationDataset(PROCESSED_DATA_DIR/TRAIN_DATA_FILE)
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

