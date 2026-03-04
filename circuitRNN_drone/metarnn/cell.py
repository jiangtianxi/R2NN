import numpy as np
import torch

from .utils import to_tensor
from . import construct

def _time_step(k_n, k_c, m_n, m_c, c_n, c_c, y1, y2, dt, coupMat, fxp):
    h1 = y1.clone()
    h1[:, fxp, :] = y1[:, fxp, :] * 0
    h2 = y2.clone()
    h2[:, fxp, :] = y2[:, fxp, :] * 0
            
    kmat, mmat, cmat = construct.construct_MKC(k_n, k_c, m_n, m_c, c_n, c_c, coupMat)
    num, _ = kmat.shape

    phi = torch.linalg.inv(mmat)

    y = torch.matmul(torch.matmul(torch.linalg.inv(torch.eye(num) + dt / 2 * torch.matmul(phi, cmat)),
                                  2 * torch.eye(num) - dt**2 * torch.matmul(phi, kmat)),
                     h1) + \
        torch.matmul(torch.matmul(torch.linalg.inv(torch.eye(num) + dt / 2 * torch.matmul(phi, cmat)),
                                  -1 * torch.eye(num) + dt / 2 * torch.matmul(phi, cmat)),
                     y2)
    return y

class WaveCell(torch.nn.Module):
    """The recurrent neural network cell implementing the scalar wave equation"""

    def __init__(self,
                 dt : float,
                 coupling):

        super().__init__()

        # Set values
        self.register_buffer("dt", to_tensor(dt))
        self.coup = coupling

    def parameters(self, recursive=True):
        for param in self.coupling.parameters():
            yield param

    def forward(self, k_n, k_c, m_n, m_c, c_n, c_c, h1, h2, coupMat, fxp):

        y = _time_step(k_n, k_c, m_n, m_c, c_n, c_c, h1, h2, self.dt, coupMat, fxp)

        return y, h1
