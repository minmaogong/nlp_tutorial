from pathlib import Path

# 路径
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "models"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 文件
RAW_DATA_FILE = 'online_shopping_10_cats.csv'
MODEL_NAME = 'google-bert/bert-base-chinese'

# 超参数
TEST_SIZE = 0.2
BATCH_SIZE = 64
SEQ_LEN = 128

LEARNING_RATE = 1e-5
EPOCHS = 20
