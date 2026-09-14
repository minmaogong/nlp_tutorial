import pandas as pd

from sklearn.model_selection import train_test_split
from config import *
from tokenizer import MyJiebaTokenizer

# 预处理函数
def preprocess():

    print("数据预处理开始...")
    # 1. 读取文件
    df = pd.read_csv(RAW_DATA_DIR/RAW_DATA_FILE).drop('cat', axis=1).dropna() # axis=1 表示按列删除
    # print(df.head())

    # 2. 划分数据集
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42, stratify=df['label']) # stratify=df['label'] 表示按标签分层抽样，保证切分后的训练集和测试集中label的比例与原数据一致

    # 3. 构建词表
    # vocab_set = set()
    # for review in tqdm(train_df['review'].tolist(), desc='构建词表'):
    #     vocab_set.update(jieba.lcut(review))
    #
    # # 增加特殊token
    # id2word = [PAD_TOKEN, UNK_TOKEN] + list(vocab_set)
    # word2id = { word:id for id, word in enumerate(id2word) }
    #
    # print("词表大小：", len(id2word))
    #
    # # 保存词表到文件
    # with open(MODEL_DIR/VOCAB_FILE, 'w', encoding='utf-8') as f:
    #     f.write( '\n'.join(id2word) )
    #
    # # 4. id化（编码）
    # def encode(text):
    #     # 分词
    #     tokens = jieba.lcut(text)
    #     # 按最大长度进行填充
    #     if len(tokens) > SEQ_LEN:
    #         tokens = tokens[:SEQ_LEN]
    #     elif len(tokens) < SEQ_LEN:
    #         tokens = tokens + [PAD_TOKEN] * (SEQ_LEN - len(tokens))
    #     # 转成id列表返回
    #     return [ word2id.get(token, word2id[UNK_TOKEN]) for token in tokens ]

    # 3. 构建词表
    MyJiebaTokenizer.build_vocab(train_df['review'].tolist(), MODEL_DIR/VOCAB_FILE)

    # 4. 创建分词器
    tokenizer = MyJiebaTokenizer.create_tokenizer(MODEL_DIR/VOCAB_FILE)

    # 5. id化（编码）
    encode = lambda text: tokenizer.encode(text, seq_len=SEQ_LEN)

    train_df['review'] = train_df['review'].apply(encode)
    test_df['review'] = test_df['review'].apply(encode)

    # 5. 保存数据集到文件
    train_df.to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE, orient='records', lines=True)
    test_df.to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE, orient='records', lines=True)

    print("数据预处理结束！")

if __name__ == '__main__':
    preprocess()
