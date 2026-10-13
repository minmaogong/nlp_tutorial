# 核心逻辑：传入一批数据，前向传播得到预测概率
import torch

from model import ReviewAnalysisModel

from config import *
from transformers import AutoTokenizer

def predict_batch(model, inputs): # inputs 是一个包含input_ids, attention_mask, token_type_ids等信息的字典
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs) # 把inputs字典解包成关键字参数传入模型的forward方法，BERT模型前向传播就需要input_ids, attention_mask, token_type_ids等信息

    # 转换成概率
    batch_proba = torch.sigmoid(outputs)
    return batch_proba.tolist()

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
    model = ReviewAnalysisModel().to(device)
    model.load_state_dict(torch.load(MODEL_DIR/BEST_MODEL)) # 加载训练好的模型参数

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
