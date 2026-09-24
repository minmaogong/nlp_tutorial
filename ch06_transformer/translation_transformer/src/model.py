import math

import torch
import torch.nn as nn
from config import *

# 自定义位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000): # max_len 表示最大序列长度，d_model表示词向量维度
        super(PositionalEncoding, self).__init__()
        # 预定义位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 逐行（对每个位置pos）进行向量预计算
        for pos in range(max_len):
            # 遍历所有可能的2i 值
            for _2i in range(0, d_model, 2):
                # 代入公式，计算对应的两个数值
                pe[pos, _2i] = math.sin( pos / (10000 ** (_2i / d_model)))
                pe[pos, _2i + 1] = math.cos( pos / (10000 ** (_2i / d_model)))

        self.register_buffer('pe', pe) # register_buffer的作用是将pe注册为模型的一个缓冲区（buffer），这样在保存和加载模型时，pe会被包含在内，但不会被视为模型的可训练参数。缓冲区通常用于存储不需要梯度更新的张量，例如位置编码矩阵。调用.to(device)时会一起迁移到设备上。

    # 前向传播，传入词向量x的形状(N, L, d_model)
    def forward(self, x):
        # 记录序列长度 L
        seq_len = x.shape[1]
        # 从预计算pe矩阵中，截取前L行
        part_pe = self.pe[: seq_len]
        # 词向量叠加位置向量，返回
        return x + part_pe

# 总模型
class TranslationModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_padding_idx, tgt_padding_idx):
        super(TranslationModel, self).__init__()
        # 定义词嵌入层
        self.src_embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=DIM_MODEL,
                                          padding_idx=src_padding_idx)
        self.tgt_embedding = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=DIM_MODEL,
                                          padding_idx=tgt_padding_idx)

        # 定义位置编码层
        self.positional_encoding = PositionalEncoding(DIM_MODEL, SEQ_LEN)

        # 定义Transformer层
        self.transformer = nn.Transformer(d_model=DIM_MODEL, nhead=NUM_HEADS, num_encoder_layers=NUM_ENCODER_LAYERS,
                                          num_decoder_layers=NUM_DECODER_LAYERS, batch_first=True)

        # 线性层
        self.linear = nn.Linear(in_features=DIM_MODEL, out_features=tgt_vocab_size)

    # 前向传播，传入N条src序列，src形状(N, S), N条tgt序列，tgt形状(N, T) 每条序列都是一个id列表
    def forward(self, src, tgt, src_key_padding_mask, tgt_mask):
        # 编码
        memory = self.encode(src, src_key_padding_mask)
        # 解码
        output = self.decode(tgt, memory, tgt_mask, src_key_padding_mask)
        return output

    # 编码方法 src形状(N, S)
    def encode(self, src, src_key_padding_mask):
        # 1. 词嵌入, 得到词向量(N, S, d_model) 每个词由d_model维词向量表示
        embedding = self.src_embedding(src)
        # 2. 位置编码，得到包含位置信息的词向量(N, S, d_model)
        input = self.positional_encoding(embedding)
        # 3. 编码器前向传播，得到上下文向量(N, S, d_model)
        memory = self.transformer.encoder(src=input, src_key_padding_mask=src_key_padding_mask) # src_key_padding_mask 标记那些位置是padding，True表示该位置是padding，前向传播时会忽略这些位置
        return memory # 返回上下文向量(N, S, d_model)

    # 解码方法 tgt形状(N, T)，memory形状(N, S, d_model)
    def decode(self, tgt, memory, tgt_mask, memory_key_padding_mask):
        # 1. 词嵌入，得到词向量(N, T, d_model)
        embedding = self.tgt_embedding(tgt)
        # 2. 位置编码，得到包含位置信息的词向量(N, T, d_model)
        input = self.positional_encoding(embedding)
        # 3. 解码器前向传播，得到输出特征向量(N, T, d_model)
        output = self.transformer.decoder(tgt=input, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        # 4. 线性层特征整合，得到词表大小的映射向量(N, T, vocab_size)
        output = self.linear(output) # (N, T, vocab_size)
        return output



if __name__ == "__main__":
    src_vocab_size = 1000
    tgt_vocab_size = 1200
    # 定义数据
    input = torch.randint(src_vocab_size, size=(BATCH_SIZE, 20))  # (batch_size, seq_length)
    input_dec = torch.randint(tgt_vocab_size, size=(BATCH_SIZE, 16))  # (batch_size, seq_length)
    # 创建模型
    model = TranslationSeq2SeqModel(src_vocab_size, tgt_vocab_size, src_padding_idx=0, tgt_padding_idx=0)
    # 前向传播
    # 编码
    encoder_output, context_vector = model.encoder(input)
    print("context_vector shape: ", context_vector.shape)
    print("encoder_output shape: ", encoder_output.shape)
    # 解码
    output, hn = model.decoder(input_dec, context_vector.unsqueeze(0), encoder_output)
    print("output shape: ", output.shape)
    print("hn shape: ", hn.shape)
