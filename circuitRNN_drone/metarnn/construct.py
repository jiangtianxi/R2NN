import torch
import math
from torch.autograd import Variable


def construct_MKC(k_n, k_c, m_n, m_c, c_n, c_c, coupMat):
    dim = k_n.shape[1] + m_n.shape[1]
    kk = torch.cat([k_c, k_n], 1)
    kk = kk.repeat(dim, 1, 1)
    cc = torch.cat([c_c, c_n], 1)
    cc = cc.repeat(dim, 1, 1)
    kmat = torch.matmul(kk, coupMat)
    cmat = torch.matmul(cc, coupMat)
    K = kmat.squeeze(1)
    C = cmat.squeeze(1)

    mv = torch.cat((m_c, m_n), 1)
    M = torch.diag(mv.squeeze(0))
    # a = K.numpy()
    return K, M, C

