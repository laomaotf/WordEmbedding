from yacs.config import CfgNode as CN

_C = CN()

_C.RAW_TEXT_DIR = ""
_C.SEGMENTED_TXT = ""
_C.STOPWORDS = "data/stopwords-master/baidu_stopwords.txt"
_C.WORD_DICT = "output/word.dict"

_C.MIN_WORD_LEN = 2
_C.MIN_WORD_FREQ = 10

_C.SKIPGRAM = CN()
_C.SKIPGRAM.SLIDING_WINDOW = 2
_C.SKIPGRAM.NEGATIVE_POOL_SIZE = 1000000
_C.SKIPGRAM.DATASET = ""


_C.SOLVER = CN()
_C.SOLVER.WEIGHTS = ""
_C.SOLVER.BATCH_SIZE = 9000
_C.SOLVER.EMBEDDING_DIMS = 128
_C.SOLVER.NEG2POS = 10
_C.SOLVER.EPOCHS = 500
_C.SOLVER.LR = 0.005
_C.SOLVER.OUTPUT_DIR = "output/models"


def get_cfg_defaults():
    return _C.clone()  # 局部变量使用形式




