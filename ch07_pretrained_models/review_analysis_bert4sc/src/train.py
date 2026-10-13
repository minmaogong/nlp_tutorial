import time

import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import *
from dataset import get_dataloader
from transformers import AutoModelForSequenceClassification # 带有分类头的预训练模型

# 训练一个轮次
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    # 按批次进行迭代
    for batch in tqdm(dataloader, desc="Training"):
        inputs = { k:v.to(device) for k, v in batch.items() } # AutoModelForSequenceClassification前向传播需要labels，所以不需要pop labels
        # 前向传播
        outputs = model(**inputs) # 将inputs解包成关键字参数传入模型的forward方法，BERT模型前向传播就需要input_ids, attention_mask, token_type_ids，labels等信息
        # 计算损失
        loss = outputs.loss # AutoModelForSequenceClassification会返回一个包含loss和logits的对象，loss是计算好的损失值
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
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    # 6. 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir=LOG_DIR/time.strftime("%Y%m%d-%H%M%S"))

    # 7. 开始训练
    min_loss = float('inf')
    for epoch in range(EPOCHS):
        this_loss = train_one_epoch(model, dataloader, optimizer, device)
        tqdm.write(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {this_loss:.4f}")

        writer.add_scalar('loss', this_loss, epoch+1)

        # 判断是否保存模型
        if this_loss < min_loss:
            min_loss = this_loss
            model.save_pretrained(MODEL_DIR)
            tqdm.write("模型保存成功！")

    writer.close()

if __name__ == "__main__":
    train()
