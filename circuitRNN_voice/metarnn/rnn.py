import torch
from . import construct

class WaveRNN(torch.nn.Module):
	def __init__(self, cell, sources, dt, fxp, srcpos, probes=[]):

		super().__init__()

		self.cell = cell
		self.dt = dt
		self.srcpos = srcpos
		self.fxp = fxp

		if type(sources) is list:
			self.sources = torch.nn.ModuleList(sources)
		else:
			self.sources = torch.nn.ModuleList([sources])

		if type(probes) is list:
			self.probes = torch.nn.ModuleList(probes)
		else:
			self.probes = torch.nn.ModuleList([probes])

	def forward(self, x, output_fields=False, if_test=False):
		"""Propagate forward in time for the length of the inputs

		Parameters
		----------
		x :
			Input sequence(s), batched in first dimension
		output_fields :
			Override flag for probe output (to get fields)
		"""

		# Hacky way of figuring out if we're on the GPU from inside the model
		device = "cuda" if next(self.parameters()).is_cuda else "cpu"

		# First dim is batch
		batch_size = x.shape[0]

		# Init hidden states
		hidden_state_shape = (batch_size,) + self.cell.coup.domain_shape
		# h1 = torch.zeros(hidden_state_shape, dtype=torch.float32)
		# h2 = torch.zeros(hidden_state_shape, dtype=torch.float32)
		h1 = torch.zeros(hidden_state_shape, dtype=torch.float32, device=device)
		h2 = torch.zeros(hidden_state_shape, dtype=torch.float32, device=device)
		y_all = []

		# Because these will not change with time we should pull them out here to avoid unnecessary calculations on each
		# tme step, dramatically reducing the memory load from backpropagation
		if if_test==False:
			k_n = self.cell.coup.k_n
			k_c = self.cell.coup.k_c     
			m_n = self.cell.coup.m_n
			m_c = self.cell.coup.m_c
			c_n = self.cell.coup.c_n
			c_c = self.cell.coup.c_c
			coupMat = self.cell.coup.coupMat
            
			lowerbound = 0.0
			mean_val = 0e-3*torch.ones(k_c.size())         
			k_c = torch.where(k_c > lowerbound, k_c, mean_val)  
			mean_val = 0e-3*torch.ones(k_n.size())
			k_n = torch.where(k_n > lowerbound, k_n, mean_val)  

			# history value: upperbound = 1.1921024e-3    
			upperbound = 9.5367e-04                             
			mean_val = upperbound*torch.ones(k_c.size())
			k_c = torch.where(k_c < upperbound, k_c, mean_val)  
			mean_val = upperbound*torch.ones(k_n.size())
			k_n = torch.where(k_n < upperbound, k_n, mean_val)  
        
			self.cell.coup.kc = k_c
			self.cell.coup.kn = k_n
		else:
			k_n = self.cell.coup.kn
			k_c = self.cell.coup.kc     
			m_n = self.cell.coup.m_n
			m_c = self.cell.coup.m_c
			c_n = self.cell.coup.c_n
			c_c = self.cell.coup.c_c
			coupMat = self.cell.coup.coupMat
            
		kmat, mmat, cmat = construct.construct_MKC(k_n, k_c, m_n, m_c, c_n, c_c, coupMat)

		num, _ = kmat.shape
		phi = torch.linalg.inv(mmat)
		deno = torch.matmul(torch.linalg.inv(torch.eye(num) + self.dt / 2 * torch.matmul(phi, cmat)), phi)

		h2[:, self.srcpos, :] = self.dt**2 * x[:, self.srcpos] / 2 / mmat[self.srcpos, self.srcpos].numpy()  
		
		# Loop through time
		for i, xi in enumerate(x.chunk(x.size(1), dim=1)):

			# Propagate the fields
			h1, h2 = self.cell(k_n, k_c, m_n, m_c, c_n, c_c, h1, h2, coupMat, self.fxp)

			# Inject source(s)
			for source in self.sources:
				h1 = source(h1, xi.squeeze(-1), deno)

			if len(self.probes) > 0 and not output_fields:
				# Measure probe(s)
				probe_values = []
				for probe in self.probes:
					probe_values.append(probe(h1))
				y_all.append(torch.stack(probe_values, dim=-1))
			else:
				# No probe, so just return the fields
				y_all.append(h1)

		# Combine outputs into a single tensor
		y = torch.stack(y_all, dim=1)

		return y.squeeze(2)
