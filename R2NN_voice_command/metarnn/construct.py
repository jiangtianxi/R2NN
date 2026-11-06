# import torch
# import math
# from torch.autograd import Variable

# def construct_MKC(k_n, k_c, m_n, m_c, c_n, c_c, coupMat):
#     dim = k_n.shape[1] + m_n.shape[1]
#     kk = torch.cat([k_c, k_n], 1)
#     kk = kk.repeat(dim, 1, 1)
#     cc = torch.cat([c_c, c_n], 1)
#     cc = cc.repeat(dim, 1, 1)
#     kmat = torch.matmul(kk, coupMat)
#     cmat = torch.matmul(cc, coupMat)
#     K = kmat.squeeze(1)
#     C = cmat.squeeze(1)

#     mv = torch.cat((m_c, m_n), 1)
#     M = torch.diag(mv.squeeze(0))
#     # a = K.numpy()
#     return K, M, C

import torch
import math
from torch.autograd import Variable

def construct_MKC(k_n, k_c, m_n, m_c, c_n, c_c, coupMat, c_g):
    """Assemble global stiffness, mass and damping matrices.

    ``coupMat`` stores the signed incidence contribution of every inter-cell
    and local-resonator connection. Multiplying it by the corresponding
    coefficient vectors produces symmetric graph-Laplacian matrices.
    ``c_g`` is the fixed viscous damping from every mass to ground.
    """
    # Keep all physical tensors on the trainable stiffness tensor's device.
    device = k_n.device
    dtype = k_n.dtype
    k_c = k_c.to(device=device, dtype=dtype)
    m_n = m_n.to(device=device, dtype=dtype)
    m_c = m_c.to(device=device, dtype=dtype)
    c_n = c_n.to(device=device, dtype=dtype)
    c_c = c_c.to(device=device, dtype=dtype)
    c_g = torch.as_tensor(c_g, device=device, dtype=dtype)
    if c_g.numel() != 1 or not torch.isfinite(c_g) or c_g < 0:
        raise ValueError("c_g must be one finite, non-negative scalar")
    coupMat = coupMat.to(device=device, dtype=dtype)

    # dim = total number of degrees of freedom
    # (main masses followed by local resonators).
    dim = k_n.shape[1] + m_n.shape[1]
    # Connection coefficients in coupMat order (couplers k_c, then local
    # springs k_n), repeated once per degree of freedom.
    kk = torch.cat([k_c, k_n], 1)
    kk = kk.repeat(dim, 1, 1)
    cc = torch.cat([c_c, c_n], 1)
    cc = cc.repeat(dim, 1, 1)
    
    # Each row of coupMat is weighted by the same connection coefficients.
    kmat = torch.matmul(kk, coupMat)
    cmat = torch.matmul(cc, coupMat)
    K = kmat.squeeze(1)
    # Connection dampers plus one damper from every mass to ground.
    C = cmat.squeeze(1) + c_g.reshape(()) * torch.eye(
        dim, device=device, dtype=dtype)

    # Diagonal mass matrix: main masses m_c, then local resonators m_n.
    mv = torch.cat((m_c, m_n), 1)
    M = torch.diag(mv.squeeze(0))
    # a = K.numpy()
    return K, M, C
