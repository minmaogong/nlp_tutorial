import torch
from config import *
from dataset import get_dataloader
from model import ReviewAnalysisModel
from predict import predict_batch
from tqdm import tqdm


# 评估逻辑，返回准确率
def evaluate(model, test_loader, device):
    model.eval()
    acc_count = 0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.tolist()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1.2 加载词表
    with open(MODEL_DIR/VOCAB_FILE, 'r', encoding='utf-8') as f:
        id2word = [ line.strip() for line in f.readlines() ]

    word2id = {word:id for id, word in enumerate(id2word)}

    # 1.3 创建模型
    model = ReviewAnalysisModel(len(id2word), word2id[PAD_TOKEN]).to(device)
    model.load_state_dict(torch.load(MODEL_DIR/BEST_MODEL))

    print("模型加载成功！")

    # 2. 获取测试集加载器
    test_loader = get_dataloader(train=False)

    # 3. 调用评估逻辑
    acc = evaluate(model, test_loader, device)

    print(f"评估结果：准确率 {acc: .6f}")

if __name__ == '__main__':
    run_evaluate()


