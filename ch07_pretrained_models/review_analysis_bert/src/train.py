import time

import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import *
from dataset import get_dataloader
from model import ReviewAnalysisModel

# 训练一个轮次
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    # 按批次进行迭代
    for batch in tqdm(dataloader, desc="Training"):
        inputs = { k:v.to(device) for k, v in batch.items() } # 此时inputs是一个字典，包含input_ids, attention_mask, token_type_ids, labels等信息
        targets = inputs.pop("labels").float() # 将labels从inputs中取出，作为目标值targets，inputs中不再包含labels
        # 前向传播
        outputs = model(**inputs) # 将inputs解包成关键字参数传入模型的forward方法，BERT模型前向传播就需要input_ids, attention_mask, token_type_ids等信息
        # 计算损失
        loss = loss_fn(outputs, targets)
        # 反向传播
        loss.backward()
        # 更新参数，计算梯度
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        # 累加损失
        total_loss += loss.item()

    return total_loss / len(dataloader)

# 训练流程
def train():
    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # 2. 获取数据加载器
    dataloader = get_dataloader(train=True)

    # 4. 创建模型
    model = ReviewAnalysisModel().to(device)

    # 5. 损失函数
    loss_fn = nn.BCEWithLogitsLoss()

    # 6. 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir=LOG_DIR/time.strftime("%Y%m%d-%H%M%S"))

    # 7. 开始训练
    min_loss = float('inf')
    for epoch in range(EPOCHS):
        this_loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        tqdm.write(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {this_loss:.4f}")

        writer.add_scalar('loss', this_loss, epoch+1)

        # 判断是否保存模型
        if this_loss < min_loss:
            min_loss = this_loss
            torch.save(model.state_dict(), MODEL_DIR/BEST_MODEL)
            tqdm.write("模型保存成功！")

    writer.close()

if __name__ == "__main__":
    train()
