# import torch

# from .utils import to_tensor


# class WaveSource(torch.nn.Module):
# 	def __init__(self, x, dt):
# 		super().__init__()
# 		self.register_buffer('x', to_tensor(x, dtype=torch.int64))
# 		self.dt = dt

# 	def forward(self, Y, X, deno):
# 		X_expanded = torch.zeros(Y.size()).detach()
# 		X_expanded[:, self.x] = X
# 		return Y + self.dt ** 2 * torch.matmul(deno, X_expanded)

import torch

from .utils import to_tensor


class WaveSource(torch.nn.Module):
    """Inject a generalized force at one lattice degree of freedom."""

    def __init__(self, x, dt):
        super().__init__()
        self.register_buffer('x', to_tensor(x, dtype=torch.int64))
        self.dt = dt

    def forward(self, Y, X, deno):
        """Add the input force ``X`` at degree of freedom ``x`` to state ``Y``.

        The contribution is ``dt**2 * deno @ f``, where ``f`` is zero except
        for ``X`` at the source node and ``deno = (I + dt/2 M^-1 C)^-1 M^-1``.
        """
        # Allocate directly on the state tensor's device and with its dtype.
        X_expanded = torch.zeros_like(Y).detach()

        X_expanded[:, self.x] = X
        return Y + self.dt ** 2 * torch.matmul(deno, X_expanded)

