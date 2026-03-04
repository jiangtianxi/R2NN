import torch

from .utils import to_tensor


class WaveSource(torch.nn.Module):
	def __init__(self, x, dt):
		super().__init__()

		# These need to be longs for advanced indexing to work
		self.register_buffer('x', to_tensor(x, dtype=torch.int64))
		self.dt = dt

	def forward(self, Y, X, deno):
		X_expanded = torch.zeros(Y.size()).detach()
		X_expanded[:, self.x] = X
		return Y + self.dt ** 2 * torch.matmul(deno, X_expanded)


