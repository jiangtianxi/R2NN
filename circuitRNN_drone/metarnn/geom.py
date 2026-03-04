from copy import deepcopy
from typing import Tuple
from torch.autograd import Variable
import numpy as np
import torch
from . import construct

from .utils import to_tensor

class Coupling(torch.nn.Module):
	def __init__(self,
				 domain_shape: Tuple,
				 row: int,
				 col: int,
				 m_n: float,
				 m_c: float,
				 k_n: float,
				 k_c: float,
				 c_n: float,
				 c_c: float,
				 coupMat: float,
				 kn: float,
				 kc:float):
		super().__init__()

		self.domain_shape = domain_shape
		# self.m_n = m_n.view(row, col)
		# self.m_c = m_c.view(row, col)
		self.m_n = m_n
		self.m_c = m_c
		self.c_n = c_n
		self.c_c = c_c
# 		self.k_c = k_c
		# self.k_n = k_n
		self.k_n = torch.nn.Parameter(k_n)
		self.k_c = torch.nn.Parameter(k_c)
		self.coupMat = coupMat
        
		self.kn = kn
		self.kc = kc

