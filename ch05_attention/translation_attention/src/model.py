import torch
import torch.nn as nn
from config import *

# 自定义Attention类
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    # 前向传播，传入解码器隐藏状态(N, L_dec, hidden_size)，编码器的输出隐藏状态(N, L_enc, hidden_size)
    def forward(self, decoder_hiddens, encoder_outputs):
        # 1. 计算注意力评分 (点积评分)，形状(N, L_dec, L_enc)
        attention_scores = torch.bmm(decoder_hiddens, encoder_outputs.transpose(1, 2)) # bbm: 两个三维矩阵相乘，得到(N, L_dec, L_enc)
        # 2. 计算注意力权重，形状(N, L_dec, L_enc)
        attention_weights = torch.softmax(attention_scores, dim=-1) # 让每一行的权重和为1
        # 3. 加权求和，得到动态上下文向量，形状(N, L_dec, hidden_size)
        context_vectors = torch.bmm(attention_weights, encoder_outputs)
        # 4. 拼接合并，形状(N. L_dec, 2*hidden_size)
        combined_vectors = torch.cat([decoder_hiddens, context_vectors], dim=-1)
        return combined_vectors



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
            bidirectional=True,
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
        # last_hidden_state = output[indices, lengths-1] # 列表索引，结果形状(N, hidden_size)
        last_hidden_state = output[indices, lengths-1, :HIDDEN_SIZE]
        first_hidden_state = output[indices, 0, HIDDEN_SIZE:]

        return output, torch.cat((last_hidden_state, first_hidden_state), dim=-1)

class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super(TranslationDecoder, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM, padding_idx=padding_idx)
        # GRU层
        self.gru = nn.GRU(
            input_size=EMBEDDING_DIM,
            hidden_size=2*HIDDEN_SIZE,
            batch_first=True,
        )
        # Attention层
        self.attention = Attention()
        # 全连接层
        self.linear = nn.Linear(in_features=4*HIDDEN_SIZE, out_features=vocab_size)

    # 前向传播
    def forward(self, x, context_vector, encoder_output):
        # 1. 传入(N, L)的数据， 得到(N, L, embedding_dim)
        embedding = self.embedding(x)
        # 2. GRU前向传播，得到output形状(N, L, hidden_size)
        output, hn = self.gru(embedding, context_vector) # context_vector 作为gru的初始隐藏状态传入
        # 3. 应用Attention机制，得到合并后的特征向量(N, L, 2*hidden_size)
        combined_vector = self.attention(output, encoder_output)
        # Linear整合特征输出，形状(N, L, vocab_size)
        output = self.linear(combined_vector)

        return output, hn

# 总模型
class TranslationSeq2SeqModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_padding_idx, tgt_padding_idx):
        super(TranslationSeq2SeqModel, self).__init__()
        self.encoder = TranslationEncoder(src_vocab_size, src_padding_idx)
        self.decoder = TranslationDecoder(tgt_vocab_size, tgt_padding_idx)


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


