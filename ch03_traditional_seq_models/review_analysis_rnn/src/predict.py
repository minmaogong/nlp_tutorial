# 核心逻辑：传入一批数据，前向传播得到预测概率
import torch

from model import ReviewAnalysisModel


def predict_batch(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    # 转换成概率
    batch_proba = torch.sigmoid(outputs)
    return batch_proba.tolist()

def predict(text):
    pass

if __name__ == "__main__":
    vocab_size = 10000
    input = torch.randint(vocab_size, size=(64, 128))
    print(input)
    # 模型
    model = ReviewAnalysisModel(vocab_size, padding_idx=0)

    # 预测
    result = predict_batch(model, input)
    print(result)

    # text = "东西很好"
    # result_proba = predict(text)
    # if result_proba > 0.5:
    #     print(f"正向 置信度：{result_proba}")
    # else:
    #     print(f"负向 置信度：{1-result_proba}")
