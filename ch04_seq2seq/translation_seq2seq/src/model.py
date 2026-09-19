import torch
import torch.nn as nn
from config import *

# 自定义编码器和解码器
class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super(TranslationEncoder, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM, padding_idx=padding_idx)
        # GRU层
        self.gru = nn.GRU(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )

    # 前向传播
    def forward(self, x):
        # 传入(N, L)的数据， 得到(N, L, embedding_dim)
        embedding = self.embedding(x)
        # GRU前向传播，得到output形状(N, L, hidden_size)
        output, _ = self.gru(embedding)

        # 提取真实最后一个时间步的输出向量，形状(N, hidden_size)
        # last_hidden_state = output[:, -1, :]
        lengths = (x != self.embedding.padding_idx).sum(dim=-1) # 计算每条数据的真实长度
        indices = torch.arange(output.shape[0])
        last_hidden_state = output[indices, lengths-1] # 列表索引

        return last_hidden_state #

class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super(TranslationDecoder, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM, padding_idx=padding_idx)
        # GRU层
        self.gru = nn.GRU(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )
        # 全连接层
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=vocab_size)

    # 前向传播
    def forward(self, x, context_vector):
        # 传入(N, L)的数据， 得到(N, L, embedding_dim)
        embedding = self.embedding(x)
        # GRU前向传播，得到output形状(N, L, hidden_size)
        output, hn = self.gru(embedding, context_vector) # context_vector 作为gru的初始隐藏状态传入

        # Linear整合特征输出，形状(N, vocab_size)
        output = self.linear(output)

        return output, hn

# 总模型
class TranslationSeq2SeqModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_padding_idx, tgt_padding_idx):
        super(TranslationSeq2SeqModel, self).__init__()
        self.encoder = TranslationEncoder(src_vocab_size, src_padding_idx)
        self.decoder = TranslationDecoder(tgt_vocab_size, tgt_padding_idx)


if __name__ == "__main__":
    src_vocab_size = 1000
    tgt_vocab_size = 1024
    # 定义数据
    input = torch.randint(src_vocab_size, size=(BATCH_SIZE, 20))  # (batch_size, seq_length)
    input_dec = torch.randint(tgt_vocab_size, size=(BATCH_SIZE, 16))  # (batch_size, seq_length)
    # 创建模型
    model = TranslationSeq2SeqModel(src_vocab_size, tgt_vocab_size, src_padding_idx=0, tgt_padding_idx=0)
    # 前向传播
    # 编码
    context_vector = model.encoder(input)
    print("context_vector shape: ", context_vector.shape)
    # 解码
    output, hn = model.decoder(input_dec, context_vector.unsqueeze(0))
    print("output shape: ", output.shape)
    print("hn shape: ", hn.shape)


