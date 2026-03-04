import torch
import numpy as np

def init_coupling_mat(row, col, k_c, k_n, m_c, m_n, c_c, c_n):

    dim = k_n.shape[1] + m_n.shape[1]
    dim_k = k_n.shape[1] + k_c.shape[1]
    coupmat = torch.zeros(dim, dim_k, dim)

    for idx in range(1, row+1):
        for jdx in range(1, col+1):
            up, left, right, down = 1, 1, 1, 1
            if idx == 1:
                up = 0
            elif idx == row:
                down = 0
            if jdx == 1:
                left = 0
            elif jdx == col:
                right = 0

            unitcell_idx = (idx - 1) * col + jdx - 1

            joint_cell_up = (idx - 2) * col + jdx - 1
            joint_cell_left = (idx - 1) * col + (jdx - 1) - 1
            joint_cell_right = (idx - 1) * col + (jdx + 1) - 1
            joint_cell_down = idx * col + jdx - 1

            joint_kc_up = (col - 1) * (idx - 1) + col * (idx - 2) + jdx - 1
            joint_kc_left = (2 * col - 1) * (idx - 1) + (jdx - 1) - 1
            joint_kc_right = (2 * col - 1) * (idx - 1) + jdx - 1
            joint_kc_down = (col - 1) * idx + col * (idx - 1) + jdx - 1

            coupmat[unitcell_idx, joint_kc_up, unitcell_idx] = 1 * up
            coupmat[unitcell_idx, joint_kc_left, unitcell_idx] = 1 * left
            coupmat[unitcell_idx, joint_kc_right, unitcell_idx] = 1 * right
            coupmat[unitcell_idx, joint_kc_down, unitcell_idx] = 1 * down

            coupmat[unitcell_idx, joint_kc_up, joint_cell_up] = -1 * up
            coupmat[unitcell_idx, joint_kc_left, joint_cell_left] = -1 * left
            coupmat[unitcell_idx, joint_kc_right, joint_cell_right] = -1 * right
            coupmat[unitcell_idx, joint_kc_down, joint_cell_down] = -1 * down

            coupmat[unitcell_idx,
                    k_c.shape[1] + unitcell_idx,
                    unitcell_idx] = 1
            coupmat[unitcell_idx,
                    k_c.shape[1] + unitcell_idx,
                    k_n.shape[1] + unitcell_idx] = -1
            coupmat[row*col + unitcell_idx,
                    k_c.shape[1] + unitcell_idx,
                    unitcell_idx] = -1
            coupmat[row*col + unitcell_idx,
                    k_c.shape[1] + unitcell_idx,
                    k_n.shape[1] + unitcell_idx] = 1

    # k1=torch.tensor([[1, -1, 0, 0],[1, 0, -1, 0],[0,0,0,0],[0,0,0,0]])
    # k2=torch.tensor([[-1, 1, 0, 0],[0, 0, 0, 0],[0,1,0,-1],[0,0,0,0]])
    # k3=torch.tensor([[0, 0, 0, 0],[-1, 0, 1, 0],[0,0,0,0],[0,0,1,-1]])
    # k4=torch.tensor([[0, 0, 0, 0],[0, 0, 0, 0],[0,-1,0,1],[0,0,-1,1]])
    # k=torch.tensor([[1,2,3,4]])
    # coup = torch.stack((k1,k2,k3,k4), 0)

    # kk = torch.cat([k_c, k_n], 1)
    # kk = kk.repeat(dim, 1, 1)
    # kmat = torch.matmul(kk, coupmat)
    # kmat = kmat.squeeze(1)
    #
    # a = kmat.numpy()

    return coupmat
