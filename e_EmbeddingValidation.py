import os,sys
import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
from collections import defaultdict

embeddings_file = "models_490_1.042914867401123_vecs.tsv"
words_file = "models_490_1.042914867401123_meta.tsv"


def load_embedding(path):
    data = []
    with open(path,'r') as f:
        for line in f:
            data.append([float(x) for x in line.strip().split('\t')])
    return np.asarray(data)

def load_meta(path):
    words = {}
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            words[line] = len(words.keys())
    return words

def WordCluster():
    def _RunCluster(embeddings, words,K=50):
        clf = KMeans(n_clusters=K).fit(embeddings)
        print("[I]Cluster Information")
        print(f"[I]inertia : {clf.inertia_}")
        print(f"[I]n_iter : {clf.n_iter_}")

        result = defaultdict(list)
        for word, label in zip(words.keys(), clf.labels_):
            result[label].append(word)
        return result


    embeddings,words = load_embedding(embeddings_file),load_meta(words_file)
    result = _RunCluster(embeddings,words)
    sizes_cluster = [(c,len(result[c])) for c in result.keys()][::-1]
    keys = sorted(sizes_cluster, key = lambda x:x[1], reverse=True)
    keys = [x[0] for x in keys]
    with open("WordClusterResult.txt",'w',encoding='utf-8') as f:
        for c in keys:
            f.write("\n")
            f.write("\n")
            f.write(f"==========={c}============\n")
            f.write(' '.join(result[c]))
            f.write("\n")
            f.write("\n")




def WordAnalogy(query_path):

    if not os.path.exists(query_path):
        return

    def _LoadQuery(path):
        queries = []
        with open(path,"r",encoding="UTF-8-sig") as f: #UTF-8-sig to remove \ufeff
            lines = f.readlines()
            lines = map(lambda x: x.strip(), lines)
            lines = filter(lambda x : x != "", lines)
            queries.extend(list(lines))
        return queries

    def _CalcCosineDistance(A,B):
        dot = np.dot(A, B.transpose())
        norm_a = np.linalg.norm(A,axis=1,keepdims=True)
        norm_b = np.linalg.norm(B,axis=1,keepdims=True)
        norm_a = np.repeat(norm_a,repeats=dot.shape[1],axis=-1)
        norm_b = np.repeat(norm_b, repeats=dot.shape[0], axis=1).transpose()
        sim = dot / (norm_a * norm_b)
        return 1 - sim


    def _CalcAnalogy(embeddings, words, queries,topK=10):
        queries_vec_valid = []
        queries_word_valid = []
        for q in queries:
            #print(q)
            q0,q1,q2 = q.split(' ')
            query_in_vocab = np.all([x in words.keys() for x in [q0,q1,q2]])
            if not query_in_vocab:
                continue
            v0,v1,v2 = embeddings[words[q0]], embeddings[words[q1]], embeddings[words[q2]]
            qv = v0 - v1 + v2
            queries_vec_valid.append(qv)
            queries_word_valid.append((q0, q1, q2))
        queries_vec_valid = np.asarray(queries_vec_valid)
        dot_query2embedding = _CalcCosineDistance(queries_vec_valid, embeddings)
        index_sorted = np.argsort(dot_query2embedding,axis=-1)
        index2word = dict(zip(words.values(), words.keys()))
        results = []
        for k in range(index_sorted.shape[0]):
            w0,w1,w2 = queries_word_valid[k]
            qw = ' '.join([w0,w1,w2])
            rw = []
            for j in range(topK):
                index = index_sorted[k][j]
                rw.append(index2word[index])
            rw = ' '.join(rw)
            results.append(
                qw + "  :  " + rw
            )
        return results


    embeddings,words, queries = load_embedding(embeddings_file),load_meta(words_file), _LoadQuery(query_path)
    result = _CalcAnalogy(embeddings,words,queries)
    with open("WordAnalogyResult.txt",'w',encoding='utf-8') as f:
        f.write('\n'.join(result))



WordCluster()
WordAnalogy("AnalogyCheck.txt")