#encoding=utf-8
import os,sys,json
import ezlog,pickle
import numpy as np
from collections import Counter
from tqdm import tqdm

from config import get_cfg_defaults

def run():

    cfg = get_cfg_defaults()
    cfg.merge_from_file("config/base.yaml")
    cfg.freeze()

    log = ezlog.EZLOG(os.path.basename(__file__))

    log.logger.info("=================restarting=========================")

    word_counter = Counter()
    ############################################
    #SCAN TXT
    with open(cfg.SEGMENTED_TXT,'rb') as f:
        #f = codecs.open(cfg['subword']['output'], 'r', encoding='utf8')
        for line in tqdm(f):
            try:
                line = line.decode('utf-8').strip()
            except Exception as e:
                print(line)
                print(e)
                continue
            article_one = line.strip().split(' ')
            if article_one == []:
                continue
            conter_one = Counter(article_one)
            word_counter += conter_one
    log.logger.info("process lines done")


    #############################################
    #GET TOP-K WORDS
    N = len(word_counter.keys())
    words_counter = word_counter.most_common(N)
    words = []
    for word,freq in words_counter:
        if freq < cfg.MIN_WORD_FREQ:
            break
        words.append( word  )



    ##############################################
    ##WORD2INT AND INT2WORD
    word2int, int2word = {}, {}
    words = list(set(words))
    for index, word in enumerate(words):
        word2int[word] = index
        int2word[index] = word
    with open(cfg.WORD_DICT,'wb') as f:
        pickle.dump((word2int, int2word, words_counter),f)
    log.logger.info("create file :{}".format(cfg.WORD_DICT))

if __name__=="__main__":
    run()

