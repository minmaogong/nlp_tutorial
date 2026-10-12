from config import *
from transformers import AutoTokenizer
from datasets import load_dataset, ClassLabel


# 预处理函数
def preprocess():

    print("数据预处理开始...")
    # 1. 读取文件
    dataset = load_dataset('csv', data_files=str(RAW_DATA_DIR/RAW_DATA_FILE))['train']
    print(dataset)

    # 2. 数据清洗，并却掉cat列
    dataset = dataset.remove_columns(['cat']).filter(lambda x: x['review'] is not None)
    print(dataset)

    # 3. 划分数据集
    dataset = dataset.cast_column('label', ClassLabel(names=['n', 'p'], num_classes=2))
    dataset_dict = dataset.train_test_split(test_size=TEST_SIZE, stratify_by_column='label') # stratify_by_column 分层抽样，表示训练集和测试集的标签分布相同，避免训练集和测试集的标签分布不均衡，导致模型训练效果不佳。
    print(dataset_dict)

    # 4. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 5. 编码转换
    def batch_encode(batch):
        inputs = tokenizer(batch['review'], padding='max_length', max_length=SEQ_LEN, truncation=True) # tokenizer()返回的是一个字典，包含input_ids, attention_mask, token_type_ids等信息
        inputs['labels'] = batch['label']
        return inputs

    dataset_dict = dataset_dict.map(batch_encode, batched=True, batch_size=BATCH_SIZE, remove_columns=['label', 'review']) # 预训练模型前向传播只需要输入input_ids, attention_mask, token_type_ids, labels等信息，所以可以删除label和review列
    print(dataset_dict)

    # 6. 保存数据集到文件
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)

    print("数据预处理结束！")

if __name__ == '__main__':
    preprocess()
