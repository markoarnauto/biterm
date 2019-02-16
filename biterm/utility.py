from itertools import combinations, chain
import numpy as np
import math
import nltk
from nltk.stem.porter import PorterStemmer

def vec_to_biterms(X):
    B_d = []
    for x in X:
        b_i = [b for b in combinations(np.nonzero(x)[0], 2)]
        B_d.append(b_i)
    return B_d


def topic_summuary(P_wz, X, V, M, verbose=True):
    res = {
        'coherence': [0] * len(P_wz),
        'top_words': [[None]] * len(P_wz)
    }
    for z, P_wzi in enumerate(P_wz):
        V_z = np.argsort(P_wzi)[:-(M + 1):-1]
        W_z = V[V_z]

        # calculate topic coherence score -> http://dirichlet.net/pdf/mimno11optimizing.pdf
        C_z = 0
        for m in range(1, M):
            for l in range(m):
                D_vmvl = np.in1d(np.nonzero(X[:,V_z[l]]), np.nonzero(X[:,V_z[m]])).sum(dtype=int) + 1
                D_vl = np.count_nonzero(X[:,V_z[l]])
                if D_vl is not 0:
                    C_z += math.log(D_vmvl / D_vl)

        res['coherence'][z] = C_z
        res['top_words'][z] = W_z
        if verbose: print('Topic {} | Coherence={:0.2f} | Top words= {}'.format(z, C_z, ' '.join(W_z)))
    return res

class Lexicon:

    def __init__(self, files):

        self._sent_category = []
        self.ps = PorterStemmer()
        for file in files:
            self._sent_category.append(set([self.ps.stem(line.strip().lower()) for line in open(file)]))


    def get_w_score(self, w):
        sent_vec = np.zeros(len(self._sent_category))
        for i, set in enumerate(self._sent_category):
            if w in set:
                sent_vec[i] = 1
        return sent_vec

    def get_d_score(self, d):
        words = nltk.word_tokenize(d.lower())

        sent_vecs = np.zeros([len(words), len(self._sent_category)])
        for i, w in enumerate(words):
            sent_vecs[i] = self.get_w_score(self.ps.stem(w))

        return sent_vecs.sum(axis=0)
