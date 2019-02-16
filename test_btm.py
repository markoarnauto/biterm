from biterm.btm import oBTM
import numpy as np
from time import time
from biterm.cbtm import oBTM as c_oBTM

class TestBTM:
    def __init__(self, K, V, theta_z, beta):
        self.K = K
        self.V = len(V)
        self.theta_z = theta_z
        self.phi_wz = np.random.dirichlet([beta] * self.V, K)

    def sample_B(self, n):
        B_ = []
        for _ in range(n):
            z = np.random.choice(self.K, 1, p=self.theta_z)[0]
            b = np.random.choice(self.V, 2, p=self.phi_wz[z], replace=False)
            B_.append((b[0], b[1]))

        return B_


def test_bBTM():

    bbtm = oBTM(K, V, alpha, beta)
    t0_batch = time()
    bbtm.fit(B_, iterations=iterations)
    t1_batch = time() - t0_batch
    print()
    print("Batch: {:0.2f} done in {:0.2f}s\n\n".format(max(bbtm.theta_z), t1_batch))

    assert max(bbtm.theta_z) > threshold
    return bbtm


def test_oBTM():

    obtm = oBTM(K, V, alpha, beta, l=1.)
    t0_online = time()
    for i in range(0, len(B_), B_chunk_size):
        B_d_ = B_[i:i + B_chunk_size]
        obtm.fit(B_d_, iterations=iterations)
    t1_online = time() - t0_online
    print()
    print("Online: {:0.2f} done in {:0.2f}s\n\n".format(max(obtm.theta_z), t1_online))

    assert max(obtm.theta_z) > threshold
    return obtm

def test_c_oBTM():

    obtm = c_oBTM(K, V, alpha, beta, l=1.)
    t0_online = time()
    for i in range(0, len(B_), B_chunk_size):
        B_d_ = B_[i:i + B_chunk_size]
        obtm.fit(B_d_, iterations=iterations)
    t1_online = time() - t0_online
    print()
    print("Online: {:0.2f} done in {:0.2f}s\n\n".format(max(obtm.theta_z), t1_online))

    assert max(obtm.theta_z) > threshold
    return obtm


if __name__ == "__main__":

    threshold = 0.9

    K = 5
    V = ['anything'] * 1000
    docs = 100

    len_B = int(docs * 5.21 * 4.21) // 2

    B_chunk_size = 50
    beta = 0.01
    alpha = 1.0
    iterations = 300

    testBTM = TestBTM(K, V, [1., 0., 0., 0., 0.], beta)
    B_ = testBTM.sample_B(len_B)
    B_ = B_[:-(len_B % docs)]
    B_ = np.array(B_).reshape(docs, -1, 2)

    #test_bBTM()
    #btm = test_oBTM()
    cbtm = test_c_oBTM()

    # test transform
    topics = cbtm.transform(B_)
    assert max(topics.mean(axis=0)) > threshold



    print("\n## Test completed ##")

