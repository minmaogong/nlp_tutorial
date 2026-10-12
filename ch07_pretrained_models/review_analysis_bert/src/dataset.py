import torch
from torch.utils.data import DataLoader
from config import *
from datasets import load_from_disk # 加载箭头arrow格式的文件

# 获取DataLoader
def get_dataloader(train=True):
    # 根据参数判断获取哪个数据集
    path = PROCESSED_DATA_DIR / ('train' if train else 'test')
    # 从文件加载数据集
    dataset = load_from_disk(path)

    # 转换为Tensor
    dataset.set_format(type='torch') # 设置数据集输出格式为PyTorch张量，确保DataLoader能正确堆叠出 [batch_size, seq_len] 形状的批次

    # 创建加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


if __name__ == '__main__':
    # 测试数据加载器
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    # 取一批数据
    for batch in train_loader:
        for key, value in batch.items():
            print(key, '->', value)
        break
    test_iter = iter(test_loader)
    batch = next(test_iter)
    print(batch['input_ids'].shape, batch['attention_mask'].shape, batch['labels'].shape)

