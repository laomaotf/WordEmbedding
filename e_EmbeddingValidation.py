import os,sys
import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
from collections import defaultdict

embeddings_file = "models_490_0.6263074278831482_vecs.tsv"
words_file = "models_490_0.6263074278831482_meta.tsv"

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

def RunCluster(embeddings, words,K=30):
    clf = KMeans(n_clusters=K).fit(embeddings)
    print("[I]Cluster Information")
    print(f"[I]inertia : {clf.inertia_}")
    print(f"[I]n_iter : {clf.n_iter_}")

    result = defaultdict(list)
    for word, label in zip(words.keys(), clf.labels_):
        result[label].append(word)
    return result


embeddings,words = load_embedding(embeddings_file),load_meta(words_file)
result = RunCluster(embeddings,words)
sizes_cluster = [(c,len(result[c])) for c in result.keys()][::-1]
keys = sorted(sizes_cluster, key = lambda x:x[1], reverse=True)
keys = [x[0] for x in keys]
with open("WordCluster.txt",'w',encoding='utf-8') as f:
    for c in keys:
        f.write("\n")
        f.write("\n")
        f.write(f"==========={c}============\n")
        f.write(' '.join(result[c]))
        f.write("\n")
        f.write("\n")






