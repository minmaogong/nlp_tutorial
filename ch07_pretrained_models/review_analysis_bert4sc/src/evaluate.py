import torch
from config import *
from dataset import get_dataloader
from transformers import AutoModelForSequenceClassification
from predict import predict_batch
from tqdm import tqdm

# 评估逻辑，返回准确率
def evaluate(model, test_loader, device):
    model.eval()
    acc_count = 0
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            targets = batch.pop("labels").tolist() # 将labels从batch中取出，作为目标值targets，batch中不再包含labels
            inputs = { k:v.to(device) for k, v in batch.items() } # 此时inputs是一个字典，包含input_ids, attention_mask, token_type_ids等信息，不包含labels
            # 前向传播（批量预测，得到概率列表）
            probas = predict_batch(model, inputs)
            # 对比预测结果和标签，统计准确数量
            for proba, target in zip(probas, targets): # zip的作用是把两个或多个序列“拉链式”配对，达到同时遍历的效果
                # 根据预测概率，得到预测标签
                result = 1 if proba > 0.5 else 0
                # 判断与真实标签是否相等
                if result == target:
                    acc_count += 1
                total_count += 1

    return acc_count / total_count

def run_evaluate():
    # 1. 准备工作，创建模型model
    # 1.1 定义设备
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # 1.3 创建模型
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device) # 这里不是加载预训练模型google-bert/bert-base-chinese，是加载我们训练好的模型参数，所以路径是MODEL_DIR，而不是MODEL_NAME

    print("模型加载成功！")

    # 2. 获取测试集加载器
    test_loader = get_dataloader(train=False)

    # 3. 调用评估逻辑
    acc = evaluate(model, test_loader, device)

    print(f"评估结果：准确率 {acc: .6f}")

if __name__ == '__main__':
    run_evaluate()


