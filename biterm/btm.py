import numpy as np
from itertools import combinations, chain
from tqdm import trange


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


    def _gibbs(self, iterations):

        Z = np.zeros(len(self.B), dtype=np.int8)
        n_wz = np.zeros((len(self.V), self.K), dtype=int)
        n_z = np.zeros(self.K, dtype=int)

        for i, b_i in enumerate(self.B):
            topic = np.random.choice(self.K, 1)[0]
            n_wz[b_i[0], topic] += 1
            n_wz[b_i[1], topic] += 1
            n_z[topic] += 1
            Z[i] = topic

        for _ in trange(iterations):
            for i, b_i in enumerate(self.B):
                n_wz[b_i[0], Z[i]] -= 1
                n_wz[b_i[1], Z[i]] -= 1
                n_z[Z[i]] -= 1
                P_w0z = (n_wz[b_i[0], :] + self.beta[b_i[0], :]) / (2 * n_z + self.beta.sum(axis=0))
                P_w1z = (n_wz[b_i[1], :] + self.beta[b_i[1], :]) / (2 * n_z + 1 + self.beta.sum(axis=0))
                P_z = (n_z + self.alpha) * P_w0z * P_w1z
                # P_z = (n_z + self.alpha) * ((n_wz[b_i[0], :] + self.beta[b_i[0], :]) * (n_wz[b_i[1], :] + self.beta[b_i[1], :]) /
                #                            (((n_wz + self.beta).sum(axis=0) + 1) * (n_wz + self.beta).sum(axis=0)))  # todo check out
                P_z = P_z / P_z.sum()
                Z[i] = np.random.choice(self.K, 1, p=P_z)
                n_wz[b_i[0], Z[i]] += 1
                n_wz[b_i[1], Z[i]] += 1
                n_z[Z[i]] += 1


        return n_z, n_wz

    def fit_transform(self, B_d, iterations):
       self.fit(B_d, iterations)
       return self.transform(B_d)

    def fit(self, B_d, iterations):
        self.B = list(chain(*B_d))
        n_z, self.nwz = self._gibbs(iterations)

        self.phi_wz = (self.nwz + self.beta) / np.array([(self.nwz + self.beta).sum(axis=0)] * len(self.V))
        self.theta_z = (n_z + self.alpha) / (n_z + self.alpha).sum()

        self.alpha += self.l * n_z
        self.beta += self.l * self.nwz


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
            s_z[i] = (self.nwz[w_d][..., None] * self.S[w_d][:, None]).sum(axis=0)

        return s_z
