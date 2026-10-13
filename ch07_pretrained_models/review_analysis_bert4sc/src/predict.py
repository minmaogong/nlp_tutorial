# 核心逻辑：传入一批数据，前向传播得到预测概率
import torch

from config import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_batch(model, inputs): # inputs 是一个包含input_ids, attention_mask, token_type_ids等信息的字典
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs) # AutoModelForSequenceClassification返回的outputs是一个包含loss和logits的对象，logits是模型的输出

    # 转换成概率
    batch_proba = torch.softmax(outputs.logits, dim=-1) # AutoModelForSequenceClassification默认是二分类，所以应该使用softmax函数将logits转换成正类和负类的概率，dim=-1表示对最后一个维度进行softmax操作，形状是(batch_size, num_classes)，num_classes=2
    return batch_proba[:, 1].tolist() # batch_proba 是一个二维张量，第一维是batch_size，第二维是负类和正类两个特征，这里我们只取正类概率，所以通过[:, 1]取出正类概率，表示取出所有行的第二列

def predict(text, model, tokenizer, device):
    # 1. 处理文本，得到输入 inputs
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=SEQ_LEN, return_tensors="pt") # tokenizer()返回的是一个字典，包含input_ids, attention_mask, token_type_ids等信息
    inputs = { k:v.to(device) for k, v in inputs.items() }

    # 2. 预测
    proba = predict_batch(model, inputs)[0]
    return proba

# 应用程序函数
def run_predict():
    # 1. 准备工作，创建模型 model
    # 1.1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # 1.2 创建分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1.3.= 创建模型
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device) # 这里不是加载预训练模型google-bert/bert-base-chinese，是加载我们训练好的模型参数，所以路径是MODEL_DIR，而不是MODEL_NAME

    print("模型加载成功！")

    # 2. 运行程序
    print("欢迎使用文本情感分析模型！输入q或者quit退出...")
    while True:
        text = input("> ")

        if text.lower() in ["q", "quit"]:
            print("欢迎下次使用。")
            break
        if text.strip() == "":
            print("请输入内容...")
            continue

        result_proba = predict(text, model, tokenizer, device)
        if result_proba > 0.5:
            print(f"正向 置信度：{result_proba}")
        else:
            print(f"负向 置信度：{1-result_proba}")


if __name__ == "__main__":
    # vocab_size = 10000
    # input = torch.randint(vocab_size, size=(64, 128))
    # print(input)
    # # 模型
    # model = ReviewAnalysisModel(vocab_size, padding_idx=0)
    #
    # # 预测
    # result = predict_batch(model, input)
    # print(result)

    # text = "东西很好"
    # result_proba = predict(text)
    # if result_proba > 0.5:
    #     print(f"正向 置信度：{result_proba}")
    # else:
    #     print(f"负向 置信度：{1-result_proba}")

    run_predict()
