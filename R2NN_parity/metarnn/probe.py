import torch

from .utils import to_tensor


class WaveProbe(torch.nn.Module):
	"""Read the displacement of one degree of freedom at every time step."""
	def __init__(self, x):
		super().__init__()

		# Need to be int64
		self.register_buffer('x', to_tensor(x, dtype=torch.int64))

	def forward(self, x):
		return x[:, self.x]

class WaveIntensityProbe(WaveProbe):
	"""Squared displacement of one degree of freedom (energy readout)."""
	def __init__(self, x):
		super().__init__(x)

	def forward(self, x):
		return super().forward(x).pow(2)
