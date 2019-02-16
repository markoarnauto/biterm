import numpy as np
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t
ctypedef np.double_t DOUBLE_t
import cython
from itertools import chain
from tqdm import trange
cdef extern from "stdlib.h":
    double drand48()

cdef int sample_mult(np.ndarray[DOUBLE_t, ndim=1] p):
    cdef int K = p.shape[0]
    for i in range(K):
        p[i] += p[i - 1]

    cdef double u = drand48()
    cdef int k = -1
    for _ in range(K):
        k += 1
        if p[k] >= u * p[K - 1]:
            break

    if k == K:
        k -= 1

    return k


class oBTM:
    """ Biterm Topic Model

        Code and naming is based on this paper http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf
        Thanks to jcapde for providing the code on https://github.com/jcapde/Biterm
    """

    def __init__(self, num_topics, V, alpha=1., beta=0.01, l=0.5):
        self.K = num_topics
        self.V = V
        self.alpha = np.full(self.K, alpha)
        self.beta = np.full((len(self.V), self.K), beta)
        self.l = l


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def _gibbs(self, int iterations, list B):

        cdef np.ndarray[DTYPE_t, ndim=1] Z = np.zeros(len(B), dtype=np.int)
        cdef np.ndarray[DTYPE_t, ndim=2] n_wz = np.zeros((len(self.V), self.K), dtype=np.int)
        cdef np.ndarray[DTYPE_t, ndim=1] n_z = np.zeros(self.K, dtype=np.int)

        for i, b_i in enumerate(B):
            topic = np.random.choice(self.K, 1)[0]
            n_wz[b_i[0], topic] += 1
            n_wz[b_i[1], topic] += 1
            n_z[topic] += 1
            Z[i] = topic

        cdef np.ndarray[DOUBLE_t, ndim=1] P_z = np.zeros(self.K, dtype=np.double)
        cdef int b_i0, b_i1, Z_iprior, Z_ipost

        for _ in trange(iterations):
            for i, b_i in enumerate(B):
                Z_iprior = Z[i]
                b_i0 = b_i[0]
                b_i1 = b_i[1]
                n_wz[b_i0, Z_iprior] -= 1
                n_wz[b_i1, Z_iprior] -= 1
                n_z[Z_iprior] -= 1
                P_w0z = (n_wz[b_i0, :] + self.beta[b_i0, :]) / (2 * n_z + self.beta.sum(axis=0))
                P_w1z = (n_wz[b_i1, :] + self.beta[b_i1, :]) / (2 * n_z + 1 + self.beta.sum(axis=0))
                P_z = (n_z + self.alpha) * P_w0z * P_w1z
                P_z = P_z / P_z.sum()

                Z_ipost = sample_mult(P_z)
                Z[i] = Z_ipost
                n_wz[b_i0, Z_ipost] += 1
                n_wz[b_i1, Z_ipost] += 1
                n_z[Z_ipost] += 1

        return n_z, n_wz

    def fit_transform(self, B_d, iterations):
       self.fit(B_d, iterations)
       return self.transform(B_d)

    def fit(self, B_d, iterations):
        B = list(chain(*B_d))
        n_z, self.n_wz = self._gibbs(iterations, B)

        self.phi_wz = (self.n_wz + self.beta) / np.array([(self.n_wz + self.beta).sum(axis=0)] * len(self.V))
        self.theta_z = (n_z + self.alpha) / (n_z + self.alpha).sum()

        self.alpha += self.l * n_z
        self.beta += self.l * self.n_wz


    def transform(self, B_d):

        P_zd = np.zeros([len(B_d), self.K])
        for i, d in enumerate(B_d):
            P_zb = np.zeros([len(d), self.K])
            for j, b in enumerate(d):
                P_zbi = self.theta_z * self.phi_wz[b[0], :] * self.phi_wz[b[1], :]
                P_zb[j] = P_zbi / P_zbi.sum()
            P_zd[i] = P_zb.sum(axis=0) / P_zb.sum(axis=0).sum()

        return P_zd


class sBTM(oBTM):

    def __init__(self, S, num_topics, V, alpha=1., beta=0.01, l=0.5):
        oBTM.__init__(self, num_topics, V, alpha, beta, l)
        self.S = S

    def transform(self, B_d):
        # P_zd = super().transform(B_d)

        s_z = np.zeros((len(B_d), self.K, self.S.shape[1]))
        for i, d in enumerate(B_d):
            w_d = list(set(chain(*d)))
            s_z[i] = (self.n_wz[w_d][..., None] * self.S[w_d][:, None]).sum(axis=0)

        return s_z

