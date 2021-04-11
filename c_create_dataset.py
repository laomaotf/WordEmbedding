#encoding=utf-8
import os,json

import sys


#import string,unicodedata

import ezlog,pickle
import numpy as np
import random,codecs
from collections import Counter
from tqdm import tqdm

from config import get_cfg_defaults

def CreateNegPool(cfg, word2int, words_freq):
    # 构造一个词频采样的集合，作为negative sampling的数据
    negative_sample_size = cfg.SKIPGRAM.NEGATIVE_POOL_SIZE
    keys = list(map(lambda x: x[0], words_freq))
    values = list(map(lambda x: x[1], words_freq))
    sampling = np.array(values) ** 0.75
    sampling_ratio = sampling / sum(sampling)
    sampling_count = np.round(negative_sample_size * sampling_ratio)
    sampling_pool = []
    for key, num in zip(keys, sampling_count):
        try:
            index = word2int[key]
            sampling_pool += [index] * int(num)
        except Exception as e:
            continue
    sampling_pool = np.array(sampling_pool)
    return sampling_pool

def run():

    cfg = get_cfg_defaults()
    cfg.merge_from_file("config/base.yaml")
    cfg.freeze()

    log = ezlog.EZLOG(os.path.basename(__file__))

    log.logger.info("=================restarting=========================")

    log.logger.info("load word dict:{}".format(cfg.WORD_DICT))
    with open(cfg.WORD_DICT, 'rb') as f:
        word2int, int2word,words_counter = pickle.load(f)
    log.logger.info("word number: {}".format(len(word2int.keys())))

    words = []

    for word in word2int.keys():
        line = "{} {}".format(word, word2int[word])
        try:
            line = line.encode('gbk')
        except Exception as e:
            print(line)
            continue
        words.append(word)

    #构造一个词频采样的集合，作为negative sampling的数据
    sampling_pool = CreateNegPool(cfg, word2int, words_counter)

    words = set(words)

    SWSize = cfg.SKIPGRAM.SLIDING_WINDOW
    data = []
    with open(cfg.SEGMENTED_TXT, 'rb') as f:
        for line in tqdm(f):
            try:
                line = line.decode('utf-8').strip()
            except Exception as e:
                print(e)
                continue
            article_one = line.strip().split(' ')
            if article_one == []:
                continue
            #过滤字典之外的词
            article_one = list(filter(lambda x: x in words, article_one))
            if len(article_one) < SWSize * 2:
                continue
            for word_index, word in enumerate(article_one):
                for nb_word in article_one[max(word_index - SWSize, 0): min(word_index + SWSize, len(article_one)) + 1]:
                    if nb_word != word:
                        data.append([word, nb_word])

    log.logger.info("process articles done")
    log.logger.info("data number: {}".format(len(data)))




    ##########################################################
    #CONVERT WORD TO WORD INDEX
    X = []  # input word
    Y = []  # output word
    for data_index, data_word in enumerate(data):
        if 0 == (data_index + 1) % 50000:
            log.logger.info("create trainval #{}".format(data_index + 1))
        if (data_word[0] in words) and (data_word[1] in words):
            X.append(word2int[data_word[0]])
            Y.append(word2int[data_word[1]])


    log.logger.info("create trainval done")
    log.logger.info("trainval : {}".format(len(X)))


    with open(cfg.SKIPGRAM.DATASET,"wb") as f:
        pickle.dump((X,Y,sampling_pool), f)



if __name__=="__main__":
    run()



