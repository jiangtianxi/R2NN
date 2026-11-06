import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def to_tensor(x, dtype=None):
	"""Convert a NumPy array or Python value to a tensor."""
	dtype = dtype if dtype is not None else torch.get_default_dtype()
	if type(x) is np.ndarray:
		return torch.from_numpy(x).type(dtype=dtype)
	else:
		return torch.tensor(x, dtype=dtype)


def set_dtype(dtype=None):
	"""Set the global default floating-point dtype."""
	if dtype == 'float32' or dtype is None:
		torch.set_default_dtype(torch.float32)
	elif dtype == 'float64':
		torch.set_default_dtype(torch.float64)
	else:
		raise ValueError('Unsupported data type: %s; should be either float32 or float64' % dtype)


def window_data(X, window_length):
	"""Window the sample, X, to a length of window_length centered at the middle of the original sample
	"""
	return X[int(len(X) / 2 - window_length / 2):int(len(X) / 2 + window_length / 2)]


def accuracy_onehot(y_pred, y_label):
	"""Compute the accuracy for a onehot
	"""
	return (y_pred.argmax(dim=1) == y_label).float().mean().item()


def energy_to_logits(X, eps=1e-12):
	"""Convert non-negative class energies to logits without changing their ratio."""
	if eps <= 0:
		raise ValueError("eps must be positive")
	return torch.log(torch.clamp_min(X, 0) + eps)


def normalize_power(X, eps=1e-12):
	"""Normalize class energies into probabilities, with a uniform zero-energy fallback."""
	if eps <= 0:
		raise ValueError("eps must be positive")

	energy = torch.clamp_min(X, 0)
	total_energy = torch.sum(energy, dim=1, keepdim=True)
	probability = energy / torch.clamp_min(total_energy, eps)
	uniform_probability = torch.full_like(energy, 1.0 / energy.shape[1])
	return torch.where(total_energy > eps, probability, uniform_probability)
