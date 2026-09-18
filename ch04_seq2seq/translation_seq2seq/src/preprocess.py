import pandas as pd

from sklearn.model_selection import train_test_split
from config import *
from tokenizer import ChineseTokenizer, EnglishTokenizer

# 预处理函数
def preprocess():

    print("数据预处理开始...")
    # 1. 读取文件
    df = pd.read_csv(RAW_DATA_DIR/RAW_DATA_FILE, sep='\t', header=None, usecols=[0, 1], names=['en', 'zh']).dropna()
    # print(df.head())

    # 2. 划分数据集
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)

    # 3. 构建词表
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), MODEL_DIR/ZH_VOCAB_FILE)
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), MODEL_DIR/EN_VOCAB_FILE)

    # 4. 创建分词器
    zh_tokenizer = ChineseTokenizer.create_tokenizer(MODEL_DIR/ZH_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.create_tokenizer(MODEL_DIR/EN_VOCAB_FILE)

    # 5. id化（编码）
    zh_encode = lambda text: zh_tokenizer.encode(text, mark=False)
    en_encode = lambda text: en_tokenizer.encode(text, mark=True)

    train_df['zh'] = train_df['zh'].apply(zh_encode)
    train_df['en'] = train_df['en'].apply(en_encode)

    test_df['zh'] = test_df['zh'].apply(zh_encode)
    test_df['en'] = test_df['en'].apply(en_encode)

    # 5. 保存数据集到文件
    train_df.to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE, orient='records', lines=True) # orient='records' 把每一行（row）当成一个独立的 JSON 对象（即一个字典） lines=True 将每条记录单独写成一行 JSON，并以换行符分隔
    test_df.to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE, orient='records', lines=True)

    print("数据预处理结束！")

if __name__ == '__main__':
    preprocess()
