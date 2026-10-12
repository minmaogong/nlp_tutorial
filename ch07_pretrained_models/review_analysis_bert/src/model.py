import torch
import torch.nn as nn
from config import *
from transformers import AutoModel

class ReviewAnalysisModel(nn.Module):
    def __init__(self):
        super(ReviewAnalysisModel, self).__init__()
        # BERT层
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        # 全连接层
        self.linear = nn.Linear(in_features=self.bert.config.hidden_size, out_features=1)

    # 前向传播
    def forward(self, **kwargs):
        # BERT，得到CLS位置对应向量 (N, hidden_size)
        output = self.bert(**kwargs)
        cls_hidden_state = output.pooler_output

        # Linear整合特征输出，形状(N, 1)
        output = self.linear(cls_hidden_state)

        return output.squeeze(-1) # (N, 1) -> (N,) 二维变1维，把第二个维度去掉

if __name__ == "__main__":
    model = ReviewAnalysisModel()
    print(model)


