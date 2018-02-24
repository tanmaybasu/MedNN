import os
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import TfidfTransformer


def load_karypis(corpus, path="/run/media/avideep/Films/Important/MSc_Project/karypis", min_df=3):
    data_file = os.path.join(path, "{}.mat".format(corpus))
    target_file = os.path.join(path, "{}.mat.rclass".format(corpus))
    
    with open(data_file) as infile:
        documents, terms = map(int, next(infile).split()[:2])
        row, col, data = [], [], []
        df_counter = Counter()
        for idx, line in enumerate(infile):
            content = line.strip()
            if content:
                it = iter(content.split())
                while True:
                    try:
                        term = int(next(it)) - 1
                        col.append(term)
                        df_counter[term] += 1
                        data.append(int(float(next(it))))
                        row.append(idx)
                    except StopIteration:
                        break
    
    counts = csc_matrix((data, (row, col)), shape=(documents, terms))
    counts = counts[:, [term for term in df_counter if df_counter[term] >= min_df]]
    tfidf = TfidfTransformer().fit_transform(counts.tocsr())
    
    with open(target_file) as infile:
        target, target_hash, topic = [], {}, 0
        for line in infile:
            content = line.strip().lower()
            if content:
                if content not in target_hash:
                    target_hash[content] = topic
                    topic += 1
                target.append(target_hash[content])
    target = np.array(target, dtype=np.int32)
    
    return tfidf, target, topic


