import numpy as np
from itertools import combinations, chain
from tqdm import trange
from .vose_sampler import VoseAlias
import random
print ('fastbtm')

class oBTM:
    """ Biterm Topic Model

        Code and naming is based on this paper http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf
        Thanks to jcapde for providing the code on https://github.com/jcapde/Biterm
    """

    def __init__(self, num_topics, V, alpha=1.0, beta=0.01, l=0.5):
        self.K = num_topics
        self.V = V
        self.alpha = np.full(self.K, alpha)
        self.beta = np.full((len(self.V), self.K), beta)
        self.l = l

    def compute_corpus_acceptance(self,n_z,n_wz,b_i,s_topic,t_topic):
        doc_proposal_1 = (n_z[t_topic] + self.alpha[t_topic])*(n_wz[b_i[0],t_topic] + self.beta[b_i[0],t_topic])*(n_wz[b_i[1],t_topic] + self.beta[b_i[1],t_topic])
        doc_proposal_2 = (n_z[s_topic] + self.alpha[s_topic])*(n_wz[b_i[0],s_topic] + self.beta[b_i[0],s_topic])*(n_wz[b_i[1],s_topic] + self.beta[b_i[1],s_topic]) 
        doc_1 = doc_proposal_1/doc_proposal_2
        doc_proposal_3 = ((2 * n_z[s_topic] + (len(self.V)*self.beta[b_i[0],s_topic]))**2)*(n_z[s_topic]+1+self.alpha[s_topic])
        doc_proposal_4 =  ((2 * n_z[t_topic] + (len(self.V)*self.beta[b_i[0],s_topic]))**2)*(n_z[t_topic]+1+self.alpha[t_topic])
        doc_2 = doc_proposal_3/doc_proposal_4
        doc_proposal = doc_1*doc_2
        return min(1,doc_proposal)

    def compute_term_proposal_1(self,wi,wj,n_wz,n_z,s_topic,t_topic):
        return (n_wz[wi,t_topic] + self.beta[wi,t_topic])*(n_wz[wj,t_topic] + self.beta[wj,t_topic])*((2 * n_z[s_topic] + (len(self.V)*self.beta[wi,s_topic]))**2)

    def compute_term_proposal_2(self,wi,wj,n_wz,n_z,s_topic,t_topic):
        return (n_wz[wi,s_topic] + self.beta[wi,s_topic])*(n_wz[wj,s_topic] + self.beta[wj,s_topic])*((2 * n_z[t_topic] + (len(self.V)*self.beta[wi,t_topic]))**2)
    
    def compute_term_proposal_3(self,wi,wj,n_wz,n_z,s_topic,t_topic):
        return (n_z[t_topic] + self.alpha[t_topic])*(n_wz[wi,s_topic] + self.beta[wi,s_topic])*(2 * n_z[t_topic] + 1 + (len(self.V)*self.beta[wi,t_topic]))

    def compute_term_proposal_4(self,wi,wj,n_wz,n_z,s_topic,t_topic):
        return (n_z[s_topic] + self.alpha[s_topic])*(n_wz[wi,t_topic] + self.beta[wi,t_topic])*(2 * n_z[s_topic] + 1 + (len(self.V)*self.beta[wi,s_topic]))
    
    def compute_term_acceptance(self,wi,wj,n_wz,n_z,s_topic,t_topic):
        t1 = self.compute_term_proposal_1(wi,wj,n_wz,n_z,s_topic,t_topic)
        t2 = self.compute_term_proposal_2(wi,wj,n_wz,n_z,s_topic,t_topic)
        t = t1/t2
        t3 = self.compute_term_proposal_3(wi,wj,n_wz,n_z,s_topic,t_topic)
        t4 = self.compute_term_proposal_4(wi,wj,n_wz,n_z,s_topic,t_topic)
        n = t3/t4
        term_proposal = t*n
        return min(1,term_proposal)

    def _gibbs(self, iterations):
        Z = np.zeros(len(self.B), dtype=np.int16)
        n_wz = np.zeros((len(self.V), self.K), dtype=int)
        n_z = np.zeros(self.K, dtype=int)
        n_aw = np.zeros(len(self.V),dtype=object)
        n_dw = np.zeros(len(self.B),dtype=int)

        for i, b_i in enumerate(self.B):
            topic = np.random.choice(self.K, 1)[0]
            n_wz[b_i[0], topic] += 1
            n_wz[b_i[1], topic] += 1
            n_z[topic] += 1
            Z[i] = topic
            n_dw[i] = topic
        
        for index,item in enumerate(n_wz):
            n_aw[index] = VoseAlias(item.tolist())
        #create alias table for each word

        for _ in trange(iterations):
            for i, b_i in enumerate(self.B):
                n_wz[b_i[0], Z[i]] -= 1
                n_wz[b_i[1], Z[i]] -= 1
                n_z[Z[i]] -= 1
                proposal = np.random.randint(0,2)
                k_topic = Z[i]
                # index = np.random.randint(0,self.K)
                # proposal_topic = n_dw[int(index)]
                # if proposal == 0:
                #     mh_acceptance = self.compute_corpus_acceptance(n_z,n_wz,b_i,k_topic, proposal_topic)
                # else:
                #     proposal_topic = n_aw[b_i[0]].alias_generation()
                #     mh_acceptance = self.compute_term_acceptance(b_i[0],b_i[1],n_wz,n_z,k_topic,proposal_topic)
                # mh_sample = random.uniform(0,1)
                # if (mh_sample < mh_acceptance):
                #     Z[i] = k_topic
                #     n_wz[b_i[0], Z[i]] += 1
                #     n_wz[b_i[1], Z[i]] += 1
                #     n_z[Z[i]] += 1
                # else:
                #     Z[i] = proposal_topic
                #     n_wz[b_i[0], Z[i]] += 1
                #     n_wz[b_i[1], Z[i]] += 1
                #     n_z[Z[i]] += 1                    
                for mh_step in range(1,2):
                    index = np.random.randint(0,self.K)
                    proposal_topic = n_dw[int(index)]
                    # proposal_topic = n_wz[b_i[0],index] #doesnt matter which biterm[0] or 1
                    mh_acceptance = self.compute_corpus_acceptance(n_z,n_wz,b_i,k_topic, proposal_topic)
                    mh_sample = random.uniform(0,1)
                    if (mh_sample < mh_acceptance):
                        k_topic = proposal_topic
                    proposal_topic = n_aw[b_i[0]].alias_generation()
                    mh_acceptance = self.compute_term_acceptance(b_i[0],b_i[1],n_wz,n_z,k_topic,proposal_topic)
                    mh_sample = random.uniform(0,1)
                    if (mh_sample < mh_acceptance):
                        k_topic = proposal_topic
                    proposal_topic = n_aw[b_i[1]].alias_generation()
                    mh_acceptance = self.compute_term_acceptance(b_i[0],b_i[1],n_wz,n_z,k_topic,proposal_topic)
                    mh_sample = random.uniform(0,1)
                    if (mh_sample < mh_acceptance):
                        k_topic = proposal_topic
                # n_aw[b_i[0]] = VoseAlias(n_wz[b_i[0]].tolist())
                # n_aw[b_i[1]] = VoseAlias(n_wz[b_i[1]].tolist())
                Z[i] = k_topic
                n_wz[b_i[0], Z[i]] += 1
                n_wz[b_i[1], Z[i]] += 1
                n_z[Z[i]] += 1
                #   
        return n_z, n_wz

    def fit_transform(self, B_d, iterations):
       self.fit(B_d, iterations)
       return self.transform(B_d)

    def fit(self, B_d, iterations):
        print ("fastbtm")
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
