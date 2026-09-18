import jieba
from tqdm import tqdm
from config import *

# 自定义分词器类
class BaseTokenizer:
    pad_token = PAD_TOKEN
    unk_token = UNK_TOKEN
    start_token = SOS_TOKEN
    end_token = EOS_TOKEN

    # 初始化方法，传入词的列表
    def __init__(self, vocab_list):
        self.vocab_size = len(vocab_list)
        # 词表
        self.id2word = vocab_list
        self.word2id = { word:id for id, word in enumerate(vocab_list) }
        # Token和id
        self.pad_id = self.word2id[self.pad_token]
        self.unk_id = self.word2id[self.unk_token]
        self.start_id = self.word2id[self.start_token]
        self.end_id = self.word2id[self.end_token]

    # 工厂方法
    @classmethod
    def create_tokenizer(cls, vocab_file):
        # 加载文件，得到词表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list =[ line.strip() for line in f.readlines() ]
        # 传入词表，构建对象返回
        return cls(vocab_list)

    # 分词
    @classmethod
    def tokenize(cls, text) -> list[str]:
        pass

    # 构建词表：传入语料，保存到文件
    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        vocab_set = set()
        for sentence in tqdm(sentences, desc="Building vocabulary"):
            vocab_set.update(cls.tokenize(sentence))

        # 增加特殊token
        id2word = [cls.pad_token, cls.unk_token, cls.start_token, cls.end_token] + list(vocab_set)

        # 保存词表到文件
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(id2word))

    # id化（编码）
    def encode(self, text, mark=False):
        # 分词
        tokens = self.tokenize(text)

        # 如果是目标语言，则需要在首尾添加<SOS><EOS> （只有在解码时才需要添加<SOS><EOS>特殊标记，编码时不需要添加）
        if mark:
            tokens = [self.start_token] + tokens + [self.end_token]

        # 转成id列表返回
        return [ self.word2id.get(token, self.unk_id) for token in tokens ]

# 分别定义中英文分词器
class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text):
        return jieba.lcut(text)

from nltk import TreebankWordTokenizer, TreebankWordDetokenizer
class EnglishTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, text) -> list[str]:
        return cls.tokenizer.tokenize(text)


    # 解码方法，传入id列表，返回英文句子
    def decode(self, ids) -> str:
        tokens = [ self.id2word[id] for id in ids ]
        return self.detokenizer.detokenize(tokens)

if __name__ == '__main__':
    # 创建一个分词器
    zh_tokenizer = ChineseTokenizer.create_tokenizer(MODEL_DIR/ZH_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.create_tokenizer(MODEL_DIR/EN_VOCAB_FILE)
    # 查看属性
    print(f"中文词表大小：{zh_tokenizer.vocab_size}")
    print(f"中文特殊符号：{zh_tokenizer.pad_id}-{zh_tokenizer.pad_token}, {zh_tokenizer.unk_id}-{zh_tokenizer.unk_token}, {zh_tokenizer.start_id}-{zh_tokenizer.start_token}, {zh_tokenizer.end_id}-{zh_tokenizer.end_token}")

    print(f"英文词表大小：{en_tokenizer.vocab_size}")

    print(zh_tokenizer.encode("我喜欢自然语言处理"))
    print(en_tokenizer.encode("Hello world! Hello NLP!", mark=True))
