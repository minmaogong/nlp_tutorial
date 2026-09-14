import jieba
from tqdm import tqdm
from config import *

# 自定义分词器类
class MyJiebaTokenizer:
    pad_token = PAD_TOKEN
    unk_token = UNK_TOKEN

    # 初始化方法，传入词的列表
    def __init__(self, vocab_list):
        self.vocab_size = len(vocab_list)
        # 词表
        self.id2word = vocab_list
        self.word2id = { word:id for id, word in enumerate(vocab_list) }
        # Token和id
        self.pad_id = self.word2id[self.pad_token]
        self.unk_id = self.word2id[self.unk_token]

    # 工厂方法
    @classmethod
    def create_tokenizer(cls, vocab_file):
        # 加载文件，得到词表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list =[ line.strip() for line in f.readlines() ]
        # 传入词表，构建对象返回
        return cls(vocab_list)

    # 分词
    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    # 构建词表：传入语料，保存到文件
    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        vocab_set = set()
        for sentence in tqdm(sentences, desc="Building vocabulary"):
            vocab_set.update(cls.tokenize(sentence))

        # 增加特殊token
        id2word = [cls.pad_token, cls.unk_token] + list(vocab_set)

        # 保存词表到文件
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(id2word))

    # id化（编码）
    def encode(self, text, seq_len):
        # 分词
        tokens = self.tokenize(text)
        # 按最大长度进行填充
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            tokens += [self.pad_token] * (seq_len - len(tokens))

        # 转成id列表返回
        return [ self.word2id.get(token, self.unk_id) for token in tokens ]

if __name__ == '__main__':
    # 创建一个分词器
    tokenizer = MyJiebaTokenizer.create_tokenizer(MODEL_DIR/VOCAB_FILE)
    # 查看属性
    print(f"词表大小：{tokenizer.vocab_size}")
    print(f"特殊符号：{tokenizer.pad_id}-{tokenizer.pad_token}, {tokenizer.unk_id}-{tokenizer.unk_token}")

    # text = "我喜欢乘坐地铁"
    text = "我喜欢乘坐宇宙飞船"
    # 分词
    tokens = tokenizer.tokenize(text)
    # id化
    ids = tokenizer.encode(text, seq_len=10)
    print(f"分词结果：{tokens}")
    print(f"id化结果：{ids}")
