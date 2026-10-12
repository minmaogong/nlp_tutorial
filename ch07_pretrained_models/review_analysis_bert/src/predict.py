# 核心逻辑：传入一批数据，前向传播得到预测概率
import jieba
import torch

from model import ReviewAnalysisModel

from config import *
from tokenizer import MyJiebaTokenizer


def predict_batch(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    # 转换成概率
    batch_proba = torch.sigmoid(outputs)
    return batch_proba.tolist()

def predict(text, model, tokenizer, device):
    # 1. 处理文本，得到输入 inputs
    # 1.1 分词
    # tokens = jieba.lcut(text)
    # 1.2 id化（编码）
    # ids = [ word2id.get(token, word2id[UNK_TOKEN]) for token in tokens ]
    # 1.1 分词并id化（编码）
    ids = tokenizer.encode(text, seq_len=SEQ_LEN)
    # 1.3 转换tensor， 形状(N=1, L)
    input = torch.tensor([ids], dtype=torch.long).to(device)

    # 2. 预测
    proba = predict_batch(model, input)[0]
    return proba

# 应用程序函数
def run_predict():
    # 1. 准备工作，创建模型 model
    # 1.1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1.2 加载词表
    # with open(MODEL_DIR/VOCAB_FILE, 'r', encoding='utf-8') as f:
    #     id2word = [line.strip() for line in f.readlines()] #line.strip() 的作用是去掉每行字符串的首尾空格和换行符
    #
    # word2id = { word:id for id, word in enumerate(id2word) }

    # 1.2 创建分词器
    tokenizer = MyJiebaTokenizer.create_tokenizer(MODEL_DIR/VOCAB_FILE)

    # 1.3.= 创建模型
    model = ReviewAnalysisModel(tokenizer.vocab_size, padding_idx=tokenizer.pad_id).to(device)
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
