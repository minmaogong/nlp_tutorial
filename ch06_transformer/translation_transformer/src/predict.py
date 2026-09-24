import torch

from model import TranslationModel

from config import *
from tokenizer import ChineseTokenizer, EnglishTokenizer


def predict_batch(model, inputs, tokenizer, device):
    model.eval()
    with torch.no_grad():
        # 编码, 得到上下文向量，形状(N, d_model)
        src_key_padding_mask = (inputs == model.src_embedding.padding_idx)
        memory = model.encode(inputs, src_key_padding_mask)
        # 解码
        # 1. 定义第一个位置输入(<SOS>)，形状(N, L=1)，N条数据，每条数据一个<sos>词id，所以是(N, 1)
        batch_size = inputs.shape[0]
        decoder_inputs = torch.full(size=[batch_size, 1], fill_value=tokenizer.start_id).to(device)
        # 2. 自回归生成
        generated_ids = [] # 只保存生成真实译文的对应id，有N个元素的列表，每个元素都是id的列表，表示一句译文
        is_finished = torch.full(size=[batch_size], fill_value=False).to(device) # 记录每条数据是否已生成天结束
        for i in range(SEQ_LEN):
            # 2.1 解码器前向传播，得到输出 (N, T, vocab_size)
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_inputs.shape[1])
            decoder_outputs = model.decode(tgt=decoder_inputs, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)

            # 2.2 贪心解码，得到形状(N, 1)
            next_token_ids = decoder_outputs[:, -1, :].argmax(dim=-1).unsqueeze(-1)

            # 2.3 更新解码器输入，进行下一个位置迭代， 得到形状(N, T+1)
            decoder_inputs = torch.cat([decoder_inputs, next_token_ids], dim=-1)

            # 3.4 保存到列表中 (目前的元素都是tensor)
            generated_ids.append(next_token_ids)

            # 3.5 结束判断
            is_finished |= (next_token_ids.squeeze(1) == tokenizer.end_id)
            if is_finished.all():
                break

        # 自回归生成结束，处理生成结果，转换为列表的列表
        # 1. 合并 L 个位置，得到(N, L)的Tensor
        generated_tensor = torch.cat(generated_ids, dim=1)
        # 2. 转为二维列表
        generated_list = generated_tensor.tolist()
        # 3. 判断eos的位置，截取真实的id列表
        for index, sentence_ids in enumerate(generated_list):
            if tokenizer.end_id in sentence_ids:
                end_index = sentence_ids.index(tokenizer.end_id)
                generated_list[index] = sentence_ids[:end_index]  # 截取到eos之前的id列表

    return generated_list

def predict(text, model, src_tokenizer, tgt_tokenizer, device):
    # 1. 处理文本，得到输入 inputs
    # 1.1 分词并id化（编码）
    ids = src_tokenizer.encode(text, mark=False)
    # 1.3 转换tensor， 形状(N=1, L)
    input = torch.tensor([ids], dtype=torch.long).to(device)

    # 2. 预测
    generated_list = predict_batch(model, input, tgt_tokenizer, device)

    # 3. 解码得到目标语言文本
    result = tgt_tokenizer.decode(generated_list[0])
    return result

# 应用程序函数
def run_predict():
    # 1. 准备工作，创建模型 model
    # 1.1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1.2 创建分词器
    zh_tokenizer = ChineseTokenizer.create_tokenizer(MODEL_DIR/ZH_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.create_tokenizer(MODEL_DIR/EN_VOCAB_FILE)

    # 1.3.= 创建模型
    model =  TranslationModel(src_vocab_size=zh_tokenizer.vocab_size, tgt_vocab_size=en_tokenizer.vocab_size, src_padding_idx=zh_tokenizer.pad_id, tgt_padding_idx=en_tokenizer.pad_id).to(device)
    model.load_state_dict(torch.load(MODEL_DIR/BEST_MODEL)) # 加载训练好的模型参数

    print("模型加载成功！")

    # 2. 运行程序
    print("欢迎使用中英翻译模型！输入q或者quit退出...")
    while True:
        text = input("中文：")

        if text.lower() in ["q", "quit"]:
            print("欢迎下次使用。")
            break
        if text.strip() == "":
            print("请输入内容...")
            continue

        result = predict(text, model, zh_tokenizer, en_tokenizer, device)
        print("英文：", result)




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
