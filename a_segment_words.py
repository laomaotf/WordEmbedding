# -*- coding: utf-8  -*-
# 将xml的wiki数据转换为text格式

import logging
import os.path

from tqdm import tqdm
import warnings
import jieba
import random,codecs
from string import punctuation
from config import get_cfg_defaults




def run():
    custom_punc = '------------ ，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥… …\n\r '
    all_punc = set(punctuation + custom_punc)
    cfg = get_cfg_defaults()
    cfg.merge_from_file("config/base.yaml")
    cfg.freeze()

    #working_folder = os.path.dirname(__file__)
    #raw_output = os.path.join(working_folder, cfg.SEGMENTED_TXT)
    raw_output = cfg.SEGMENTED_TXT
    if not os.path.exists( os.path.dirname(raw_output)):
        os.makedirs(os.path.dirname(raw_output))

    program = os.path.basename(__file__)  # 得到文件名
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    #################################################
    #LOADING STOPWORDS
    stopwords = []
    f = codecs.open(cfg.STOPWORDS, 'r', encoding='utf8')
    for line in f:
        stopwords.append(line.strip())
    stopwords = set(stopwords)
    all_punc = all_punc | stopwords

    bar = tqdm(os.listdir(os.path.join(cfg.RAW_TEXT_DIR)))
    lines = []
    for txt in bar:
        bar.set_description(txt)
        raw_input = os.path.join( cfg.RAW_TEXT_DIR,txt )
        with open(raw_input,'r',encoding='utf-8') as f:
            for line in f:
                words = jieba.lcut(line) #分词
                words = filter(lambda x: x not in all_punc, words) #停用词和标点符号
                words = filter(lambda x: len(x) >= cfg.MIN_WORD_LEN, words) #单词最小长度
                words = list(words)
                if words == []:
                    continue
                lines.append(' '.join(words))

    with open(raw_output,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info("Finished Saved articles.")


if __name__ == "__main__":
    run()