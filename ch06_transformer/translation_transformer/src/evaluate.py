import torch
from config import *
from dataset import get_dataloader
from model import TranslationSeq2SeqModel
from predict import predict_batch
from tqdm import tqdm
from tokenizer import ChineseTokenizer, EnglishTokenizer
from nltk.translate.bleu_score import corpus_bleu


# 评估逻辑，返回准确率
def evaluate(model, test_loader, tokenizer, device):
    model.eval()
    # 用列表保存预测结果和目标结果
    predictions = []
    references = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.tolist()
            targets = [ [target[1:target.index(tokenizer.end_id)]] for target in targets ]
            # 前向传播（批量预测，得到概率列表）
            generated_lists = predict_batch(model, inputs, tokenizer, device)
            # 将预测结果添加到列表
            predictions.extend(generated_lists)
            # 将目标添加到列表
            references.extend(targets)
    return corpus_bleu(list_of_references=references, hypotheses=predictions)

def run_evaluate():
    # 1. 准备工作，创建模型model
    # 1.1 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1.2 创建分词器
    zh_tokenizer = ChineseTokenizer.create_tokenizer(MODEL_DIR/ZH_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.create_tokenizer(MODEL_DIR/EN_VOCAB_FILE)

    # 1.3 创建模型
    model = TranslationSeq2SeqModel(zh_tokenizer.vocab_size, en_tokenizer.vocab_size, zh_tokenizer.pad_id, en_tokenizer.pad_id).to(device)
    model.load_state_dict(torch.load(MODEL_DIR/BEST_MODEL))

    print("模型加载成功！")

    # 2. 获取测试集加载器
    test_loader = get_dataloader(train=False)

    # 3. 调用评估逻辑
    bleu = evaluate(model, test_loader, en_tokenizer, device)

    print(f"评估结果：BLEU {bleu: .6f}")

if __name__ == '__main__':
    run_evaluate()


