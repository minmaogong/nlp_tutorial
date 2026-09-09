import torch
import torch.nn as nn

class ReviewAnalysisModel(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super(ReviewAnalysisModel, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=128, padding_idx=padding_idx)
        # RNN层
        self.rnn = nn.RNN(
            input_size=128,
            hidden_size=256,
            batch_first=True,
        )
        # 全连接层
        self.linear = nn.Linear(in_features=256, out_features=1)

    # 前向传播
    def forward(self, x):
        # 传入(N, L)的数据， 得到(N, L, embedding_dim)
        embedding = self.embedding(x)
        # RNN前向传播，得到output形状(N, L, hidden_size)
        output, _ = self.rnn(embedding)

        # 提取真实最后一个时间步的输出向量，形状(N, hidden_size)
        # last_hidden_state = output[:, -1, :]
        lengths = (x != self.embedding.padding_idx).sum(dim=-1) # 计算每条数据的真实长度
        indices = torch.arange(output.shape[0])
        last_hidden_state = output[indices, lengths-1] # 列表索引

        # Linear整合特征输出，形状(N, 1)
        output = self.linear(last_hidden_state)

        return output.squeeze(-1)

if __name__ == "__main__":
    vocab_size = 10000
    # 定义数据
    input = torch.randint(vocab_size, size=(64, 128))  # (batch_size, seq_length)
    # 创建模型
    model = ReviewAnalysisModel(vocab_size, padding_idx=0)
    # 前向传播
    output = model(input)
    print(output.shape)


