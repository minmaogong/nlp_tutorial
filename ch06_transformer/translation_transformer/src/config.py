from pathlib import Path

# 路径
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "models"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 文件
RAW_DATA_FILE = 'cmn.txt'
TRAIN_DATA_FILE = 'train.jsonl'
TEST_DATA_FILE = 'test.jsonl'

ZH_VOCAB_FILE = 'zh_vocab.txt'
EN_VOCAB_FILE = 'en_vocab.txt'
BEST_MODEL = 'best_model.pt'

# 特殊Token
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'

# 超参数
TEST_SIZE = 0.2
BATCH_SIZE = 64
SEQ_LEN = 128

DIM_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

LEARNING_RATE = 1e-3
EPOCHS = 50
