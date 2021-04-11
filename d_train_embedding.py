#encoding = utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras as tfks
import ezlog,os,sys,pickle,random
import json,gzip
from tqdm import tqdm
import numpy as np
import time
from config import get_cfg_defaults


print(tf.__version__)


cfg = get_cfg_defaults()
cfg.merge_from_file("config/base.yaml")
cfg.freeze()

os.makedirs(cfg.SOLVER.OUTPUT_DIR,exist_ok=True)


LOG = ezlog.EZLOG(os.path.basename(__file__))



# function to convert numbers to one hot vectors
def to_one_hot(vocab_size,data_point_index):
    temp = np.zeros((vocab_size,))
    temp[data_point_index] = 1
    return temp

class TRAINVAL_DATA:
    def __init__(self, cvt2bin=True):

        with open(cfg.WORD_DICT, 'rb') as f:
            self.word2int, self.int2word,_ = pickle.load(f)

        with open(cfg.SKIPGRAM.DATASET, "rb") as f:
            X,Y,self.sampling_pool = pickle.load(f,encoding='latin1')

        if cvt2bin:
            cache_path = os.path.join(cfg.SOLVER.OUTPUT_DIR,"trainval.cache")
            if os.path.exists(cache_path):
                with open(cache_path,'rb') as f:
                    self.train_data = pickle.load(f)
            else:
                trainval = []
                XY = []
                for x,y in zip(X,Y):
                    xy = '{},{}'.format(x,y)
                    XY.append(xy)
                    trainval.append((x,y))
                del X
                del Y
                #random.shuffle(trainval)
                self.train_data = trainval
                with open(cache_path,'wb') as f:
                    pickle.dump(self.train_data,f)
        return

class EmbeddingLayerClass(tfks.Model):
    def __init__(self,vocab_size, embedding_dim=128, predict_only = False):
        super(EmbeddingLayerClass, self).__init__()
        self.embedding = tfks.layers.Embedding(vocab_size,embedding_dim,input_length=1, name="emb")
        #single embedding for y
        self.embedding_y = tfks.layers.Embedding(vocab_size, embedding_dim, input_length=1, name="emby")
    def call(self,x, y = None, pos=True):
        x_emb = self.embedding(x)
        # if y is None:
        #     y = x
        if not y is None:
            y_emb = self.embedding_y(y)
            if pos:
                dot = tf.reduce_sum(tf.multiply(x_emb,y_emb),axis=-1)
            else:
                x_emb = tf.tile(x_emb, (tf.shape(y_emb)[0] / tf.shape(x_emb)[0],1))
                dot = tf.reduce_sum(tf.multiply(-x_emb, y_emb),axis=-1)
            #output = tf.sigmoid(dot)
            return tf.math.log_sigmoid(dot)
        return x_emb



def show_neighbour_words(model, trainval,topK = 8, test_num = 4):
    def _get_sim(model, trainval, vocab_size, valid_word_idx,batch_size = 2000):
        sim = np.zeros((vocab_size,)) - 1
        valid_word_emb = model(valid_word_index).numpy()

        input_array = np.zeros((batch_size,))
        #in_arr2 = np.zeros((batch_size,))
        keys = trainval.int2word.keys()
        for start in tqdm(range(0, len(keys), batch_size ),desc='testing scaning...'):
            #if start + batch_size >= len(keys):
            #    continue
            batch_size = len(keys) - start
            if batch_size  < 1:
                continue
            input_array = np.zeros((batch_size,))
            for index in range(start, start + batch_size):
                input_array[index-start,] = index #word to find neighbours
                #in_arr2[index-start,] = index
            #out = model.predict_on_batch([in_arr1, in_arr2])
            output = model(input_array).numpy()
            for index in range(start, start + batch_size):
                #sim[index] =  output[index - start]
                sim[index] = np.dot(output[index-start],valid_word_emb)
        return sim

    vocab_size = len(trainval.word2int.keys())
    keys = list(trainval.int2word.keys())
    random.shuffle(keys)
    keys = keys[0:test_num] + [100,200,300,400,500,600]
    for valid_word_index in keys:
        sim = _get_sim(model, trainval, vocab_size, valid_word_index)
        nearest = (-sim).argsort()[1:topK + 1]
        log_str = 'Nearest to {}:'.format(trainval.int2word[valid_word_index])
        for k in range(topK):
            if nearest[k] < 0:
                continue
            close_word = trainval.int2word[nearest[k]]
            log_str = '{} {},'.format(log_str, close_word)
        LOG.logger.info(log_str)




trainval = TRAINVAL_DATA()
model = EmbeddingLayerClass(len(trainval.word2int.keys()), embedding_dim=cfg.SOLVER.EMBEDDING_DIMS)

import io
if cfg.SOLVER.WEIGHTS  != "":
    model.build(input_shape=(1,))
    model.load_weights(cfg.SOLVER.WEIGHTS)
    LOG.logger.info(model.summary())
    emb = model.layers[0]
    emb_matrix = emb.get_weights()[0]
    print(emb_matrix.shape)

    #SAVE EMBEDDING TO SHOW IN http://projector.tensorflow.org/
    prefix_out = os.path.splitext(cfg.SOLVER.WEIGHTS)[0]
    out_v = io.open('{}_vecs.tsv'.format(prefix_out), 'w', encoding='utf-8')
    out_m = io.open('{}_meta.tsv'.format(prefix_out), 'w', encoding='utf-8')
    for index in trainval.int2word.keys():
        vec = emb_matrix[index]
        word = trainval.int2word[index]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_v.close()
    out_m.close()


class CosineScheduleClass(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,base_lr,steps_total):
        super(CosineScheduleClass, self).__init__()
        self.base_lr = base_lr
        self.steps_total = steps_total
        self.lr = base_lr
        self.min_lr = 1e-6

    def __call__(self, step):
        lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + tf.math.cos(float(step) * 3.1416/self.steps_total))
        #print("NEW LR: {} at {}".format(lr,step))
        return lr

batch_size = cfg.SOLVER.BATCH_SIZE
vocab_size = len(trainval.word2int.keys())
total_trainval = len(trainval.train_data)
steps_total = (cfg.SOLVER.EPOCHS * total_trainval) / batch_size
neg2pos = cfg.SOLVER.NEG2POS
lr = CosineScheduleClass(base_lr=cfg.SOLVER.LR,steps_total=steps_total )
opt = tfks.optimizers.Adam(lr)
LOG.logger.info("batch size {} total samples {}".format(batch_size,total_trainval))

for epoch in range(cfg.SOLVER.EPOCHS):
    random.shuffle(trainval.train_data)
    loss_hist = []
    bar = tqdm(enumerate(range(0,total_trainval,batch_size)))
    for iter,start in bar:
        centers,contents = [],[]
        for index in range(start, start + batch_size):
            if index >= len(trainval.train_data):
                break
            pos = trainval.train_data[index]
            centers.append(pos[0])
            contents.append(pos[1])
        if centers == []:
            break
        centers_num = len(centers)
        n = centers_num * neg2pos
        negs = np.random.choice(trainval.sampling_pool,n) #PICK NEGS RANDOMLY
        centers,contents = np.asarray(centers), np.asarray(contents)
        with tf.GradientTape() as tape:
            loss_pos = model(centers,y = contents,pos=True)
            loss_neg = model(centers, y = negs, pos=False)
            loss = -1 * (tf.reduce_mean(loss_pos) + tf.reduce_mean(loss_neg))
        grads = tape.gradient(loss,model.trainable_variables)
        opt.apply_gradients(zip(grads,model.trainable_variables))
        current_lr = opt._decayed_lr(tf.float32)
        loss_hist.append(loss.numpy().mean())
        bar.set_description_str("iter {} loss {}".format(iter, loss_hist[-1]))
    epoch_loss = np.asarray(loss_hist).mean()
    LOG.logger.info("epoch {} loss {}".format(epoch,epoch_loss))
    if epoch % 10 == 0:
        #show_neighbour_words(model, trainval, topK=8, test_num=8)
        model.save_weights(os.path.join(cfg.SOLVER.OUTPUT_DIR,"models_{}_{}.h5".format(epoch,epoch_loss )))

show_neighbour_words(model, trainval, topK=8, test_num=100)


