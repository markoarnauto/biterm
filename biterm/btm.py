import numpy as np
from itertools import combinations, chain
from tqdm import trange
from .vose_sampler import VoseAlias


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

    def compute_corpus_acceptance(k_topic,proposal_topic):
        doc_proposal_1 = ((n_z[k_topic] + self.alpha[k_topic])*(n_wz[b_i[0],k_topic] + self.beta[b_i[0],k_topic])*(n_wz[b_i[1],k_topic] + self.beta[b_i[0],k_topic]))/
                       (n_z[proposal_topic] + self.alpha[proposal_topic])*(n_wz[b_i[0],proposal_topic] + self.beta[b_i[0],proposal_topic])*(n_wz[b_i[1],proposal_topic] + self.beta[b_i[0],proposal_topic]) 
        doc_proposal_2 = ((2 * n_z[proposal_topic] + self.beta[proposal_topic].sum(axis=0))**2)*(n_z[proposal_topic]+1+self.alpha[proposal_topic])/
                        ((2 * n_z[k_topic] + self.beta[k_topic].sum(axis=0))**2)*(n_z[k_topic]+1+self.alpha[k_topic])
        doc_proposal = doc_proposal_1*doc_proposal_2
        return min(1,doc_proposal)

    def compute_term_acceptance(wi,wj,s_topic,t_topic):
        term_proposal_1 = (n_wz[wi,t_topic] + self.beta[wi,t_topic])*(n_wz[wj,t_topic] + self.beta[wj,t_topic])*((2 * n_z[s_topic] + self.beta[wi,s_topic])**2)/
                          (n_wz[wi,s_topic] + self.beta[wi,s_topic])*(n_wz[wj,s_topic] + self.beta[wj,s_topic])*((2 * n_z[t_topic] + self.beta[wi,t_topic])**2)
        term_proposal_2 = (n_z[t_topic] + self.alpha[t_topic])*(n_wz[wi,s_topic] + self.beta[wi,s_topic])*(2 * n_z[t_topic] + 1 + self.beta[wi,t_topic])/
                            (n_z[s_topic] + self.alpha[s_topic])*(n_wz[wi,t_topic] + self.beta[wi,t_topic])*(2 * n_z[s_topic] + 1 + self.beta[wi,s_topic])
        term_proposal = term_proposal*term_proposal_1
        return min(1,term_proposal)

    def _gibbs(self, iterations):
        Z = np.zeros(len(self.B), dtype=np.int16)
        n_wz = np.zeros((len(self.V), self.K), dtype=int)
        n_z = np.zeros(self.K, dtype=int)
        n_aw = np.zeros(self.V)

        for i, b_i in enumerate(self.B):
            topic = np.random.choice(self.K, 1)[0]
            n_wz[b_i[0], topic] += 1
            n_wz[b_i[1], topic] += 1
            n_z[topic] += 1
            Z[i] = topic
            n_aw[b_i[0]] = VoseAlias(n_wz[b_i[0]])
            n_aw[b_i[1]] = VoseAlias(n_wz[b_i[1]])
        
        #create alias table for each word
        
        n_aw[b_i[0]] = VA.n_wz[b_i[0]]

        for _ in trange(iterations):
            for i, b_i in enumerate(self.B):
                n_wz[b_i[0], Z[i]] -= 1
                n_wz[b_i[1], Z[i]] -= 1
                n_z[Z[i]] -= 1
                proposal = np.random.randint(0,1)
                k_topic = Z[i]
                if proposal == 0:
                    index = randomInt(0, len(self.V))
                    proposal_topic = n_wz[b_i[0],index] #doesnt matter which biterm[0] or 1
                    mh_acceptance = compute_corpus_acceptance(k_topic, proposal_topic)
                else :
                    proposal_topic = n_aw[b_i[0]].alias_generation()
                    mh_acceptance = compute_term_acceptance(b_i[0],b_i[1],k_topic,proposal_topic)
                mh_sample = randomFloat(0, 1)
                if (mh_sample < mh_acceptance):
                  # increment_count_matrices(d, w, k)  // reject proposal, revert to k
                    Z[i] = k
                    n_wz[b_i[0], Z[i]] += 1
                    n_wz[b_i[1], Z[i]] += 1
                    n_z[Z[i]] += 1
                else :
                    # increment_count_matrices(d, w, p)  // accept proposal
                    Z[i] = p
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
