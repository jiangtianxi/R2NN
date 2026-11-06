from copy import deepcopy
from typing import Tuple
from torch.autograd import Variable
import numpy as np
import torch
from . import construct

from .utils import to_tensor


def default_node_ids(row: int, col: int):
    """Return the default source, support and readout IDs for a grid.

    Node numbering is row-major. Main-mass IDs occupy ``[0, row*col)`` and
    local-resonator IDs occupy ``[row*col, 2*row*col)``.

    The default geometry follows the layout used by the experiments:

    * source: middle node of the top main-mass boundary;
    * fixed nodes: four main-mass corners;
    * probes: left and right local resonators on the middle row.

    For an even number of rows or columns, the lower-index central row or
    column is selected. Grids smaller than 3x3 are rejected because their
    top-middle source would overlap a corner support.
    """
    if not isinstance(row, int) or not isinstance(col, int):
        raise TypeError("row and col must be integers")
    if row < 3 or col < 3:
        raise ValueError("row and col must both be at least 3")

    n_main = row * col
    middle_column = (col - 1) // 2
    middle_row = (row - 1) // 2
    middle_left = middle_row * col
    middle_right = middle_left + col - 1

    return {
        "source": middle_column,
        "fixed": (
            0,
            col - 1,
            (row - 1) * col,
            n_main - 1,
        ),
        "probes": (
            n_main + middle_left,
            n_main + middle_right,
        ),
    }


class Coupling(torch.nn.Module):
    """Physical parameters and topology of the mass-spring lattice.

    Trainable parameters, each with shape ``(1, n)``:

    * ``k_n``: local-resonator springs, one per unit cell;
    * ``k_c``: inter-cell couplers, one per lattice edge;
    * ``k_d``: unilateral (one-sided) local springs, used only in
      ``"unilateral"`` mode.

    Fixed buffers:

    * ``m_c``, ``m_n``: main-mass and local-resonator masses;
    * ``c_c``, ``c_n``: coupler and local-spring dampers;
    * ``c_g``: damper from every mass to ground;
    * ``k_cubic``: cubic local-spring coefficients (``"cubic"`` mode);
    * ``coupMat``: incidence tensor from ``init_coupling_mat``;
    * ``kn``, ``kc``: copies of reference stiffnesses (not used in the
      dynamics).
    """

    _PHYSICAL_BUFFER_NAMES = (
        "m_n", "m_c", "c_n", "c_c", "c_g", "k_cubic",
        "coupMat", "kn", "kc")

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
                 kc:float,
                 k_d:float,
                 c_g:float = 1e-4,
                 k_cubic=None):
        super().__init__()

        self.domain_shape = domain_shape
        self.register_buffer("m_n", torch.as_tensor(m_n))
        self.register_buffer("m_c", torch.as_tensor(m_c))
        self.register_buffer("c_n", torch.as_tensor(c_n))
        self.register_buffer("c_c", torch.as_tensor(c_c))
        self.register_buffer(
            "c_g",
            torch.as_tensor(
                c_g, device=self.m_n.device, dtype=self.m_n.dtype))
        if k_cubic is None:
            k_cubic = torch.zeros_like(k_n)
        self.register_buffer(
            "k_cubic",
            torch.as_tensor(
                k_cubic, device=self.m_n.device, dtype=self.m_n.dtype))
        self.register_buffer("coupMat", torch.as_tensor(coupMat))
        self.register_buffer("kn", torch.as_tensor(kn))
        self.register_buffer("kc", torch.as_tensor(kc))

        self.k_n = torch.nn.Parameter(k_n)
        self.k_c = torch.nn.Parameter(k_c)
        self.k_d = torch.nn.Parameter(k_d)

    def __setstate__(self, state):
        """Migrate physical tensors stored by checkpoints from older releases."""
        super().__setstate__(state)
        for name in self._PHYSICAL_BUFFER_NAMES:
            legacy_value = self.__dict__.pop(name, None)
            if legacy_value is not None and name not in self._buffers:
                self.register_buffer(name, legacy_value)
        if "c_g" not in self._buffers:
            # Preserve the original dynamics of checkpoints trained before
            # ground damping was introduced.
            self.register_buffer("c_g", self.m_n.new_tensor(0.0))
        if "k_cubic" not in self._buffers:
            self.register_buffer(
                "k_cubic", torch.zeros_like(self.k_n.detach()))

    @torch.no_grad()
    def project_stiffness_(self, min_value=1e-8, dt=None,
                           max_dt2_eigenvalue=None,
                           include_nonlinear=True,
                           fixed_indices=None,
                           linear_zero_threshold=None):
        """Project stiffnesses to the physical and integration domains.

        When ``linear_zero_threshold`` is provided, entries of ``k_n`` and
        ``k_c`` strictly below it are pruned to exact zero. The nonlinear
        stiffness keeps the positive ``min_value`` floor.

        When ``dt`` and ``max_dt2_eigenvalue`` are given, all stiffnesses
        are scaled down together if ``dt**2 * lambda_max`` exceeds
        ``max_dt2_eigenvalue``. ``lambda_max`` is the largest eigenvalue of
        ``M^-1/2 (K + K_d) M^-1/2`` over the free degrees of freedom; the
        undamped central-difference scheme is stable for values below 4.
        Returns ``{"dt2_lambda_max", "projection_scale"}``.
        """
        if min_value < 0:
            raise ValueError("min_value must be non-negative")
        if (linear_zero_threshold is not None
                and linear_zero_threshold < 0):
            raise ValueError("linear_zero_threshold must be non-negative")
        if max_dt2_eigenvalue is not None and max_dt2_eigenvalue <= 0:
            raise ValueError("max_dt2_eigenvalue must be positive")

        linear_stiffnesses = (self.k_n, self.k_c)
        nonlinear_stiffnesses = (
            (self.k_d,) if include_nonlinear else ())
        stiffnesses = linear_stiffnesses + nonlinear_stiffnesses
        if not all(torch.isfinite(value).all() for value in stiffnesses):
            raise FloatingPointError("Non-finite trainable stiffness detected")

        for value in linear_stiffnesses:
            if linear_zero_threshold is None:
                value.clamp_(min=min_value)
            else:
                value.masked_fill_(value < linear_zero_threshold, 0.0)
        for value in nonlinear_stiffnesses:
            value.clamp_(min=min_value)

        stats = {
            "dt2_lambda_max": float("nan"),
            "projection_scale": 1.0,
        }
        if max_dt2_eigenvalue is None:
            return stats
        if dt is None:
            raise ValueError("dt is required for the stability projection")

        kmat, mmat, _ = construct.construct_MKC(
            self.k_n, self.k_c, self.m_n, self.m_c,
            self.c_n, self.c_c, self.coupMat, self.c_g)

        kdelta = torch.zeros_like(kmat)
        if include_nonlinear:
            n_local = self.k_d.shape[1]
            idx_main = torch.arange(n_local, device=kmat.device)
            idx_local = idx_main + n_local
            kd = self.k_d.squeeze(0)
            kdelta[idx_main, idx_main] = kd
            kdelta[idx_local, idx_local] = kd
            kdelta[idx_main, idx_local] = -kd
            kdelta[idx_local, idx_main] = -kd

        inverse_sqrt_mass = torch.rsqrt(torch.diagonal(mmat))
        normalized_stiffness = (
            inverse_sqrt_mass[:, None]
            * (kmat + kdelta)
            * inverse_sqrt_mass[None, :]
        )
        normalized_stiffness = (
            normalized_stiffness + normalized_stiffness.transpose(0, 1)) / 2
        if fixed_indices is not None:
            fixed_indices = torch.as_tensor(
                fixed_indices, device=kmat.device,
                dtype=torch.long).reshape(-1)
            free_mask = torch.ones(
                normalized_stiffness.shape[0],
                device=kmat.device, dtype=torch.bool)
            free_mask[fixed_indices] = False
            free_indices = torch.arange(
                normalized_stiffness.shape[0],
                device=kmat.device)[free_mask]
            normalized_stiffness = normalized_stiffness.index_select(
                0, free_indices).index_select(1, free_indices)
        lambda_max = torch.linalg.eigvalsh(normalized_stiffness).max()
        dt_tensor = torch.as_tensor(
            dt, device=kmat.device, dtype=kmat.dtype)
        dt2_lambda_max = dt_tensor.square() * lambda_max

        if not torch.isfinite(dt2_lambda_max):
            raise FloatingPointError(
                "Non-finite eigenvalue in stiffness stability check")

        projection_scale = torch.ones_like(dt2_lambda_max)
        if dt2_lambda_max > max_dt2_eigenvalue:
            projection_scale = (
                torch.as_tensor(
                    max_dt2_eigenvalue,
                    device=kmat.device,
                    dtype=kmat.dtype)
                / dt2_lambda_max
            )
            for value in stiffnesses:
                value.mul_(projection_scale)
            for value in linear_stiffnesses:
                if linear_zero_threshold is None:
                    value.clamp_(min=min_value)
                else:
                    value.masked_fill_(
                        value < linear_zero_threshold, 0.0)
            for value in nonlinear_stiffnesses:
                value.clamp_(min=min_value)
            dt2_lambda_max = dt2_lambda_max * projection_scale

        stats["dt2_lambda_max"] = dt2_lambda_max.item()
        stats["projection_scale"] = projection_scale.item()
        return stats

