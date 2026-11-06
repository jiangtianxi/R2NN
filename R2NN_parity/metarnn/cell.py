import numpy as np
import torch

from .utils import to_tensor
# from . import construct


VALID_NONLINEARITY_MODES = {"linear", "unilateral", "cubic"}


def build_cubic_force_local_resonators(u, k_cubic):
    """Return paired cubic restoring forces for local resonators.

    For ``delta = u_main - u_local``, each colocated pair receives
    ``[k_cubic * delta**3, -k_cubic * delta**3]``. ``k_cubic`` may be a
    scalar or one fixed coefficient per resonator.
    """
    if u.dim() == 3 and u.shape[-1] == 1:
        displacement = u.squeeze(-1)
    elif u.dim() == 2:
        displacement = u
    else:
        raise ValueError(
            "u must have shape (batch, ndof) or (batch, ndof, 1)")

    n_total = displacement.shape[1]
    if n_total % 2:
        raise ValueError(
            "The number of degrees of freedom must be even")
    n_main = n_total // 2
    relative_displacement = (
        displacement[:, :n_main] - displacement[:, n_main:])
    coefficient = torch.as_tensor(
        k_cubic,
        device=displacement.device,
        dtype=displacement.dtype)
    main_force = coefficient * relative_displacement.pow(3)
    return torch.cat((main_force, -main_force), dim=1).unsqueeze(-1)


def _time_step(T1, T2, M_const, kmat, y1, y2, fxp, k_d,
               switch_temperature=1.0, use_surrogate_gradient=False,
               use_nonlinear=True, nonlinearity_mode=None,
               k_cubic=None, state_limit=5000.0):
    """Advance the lattice by one central-difference time step.

    ``y1`` and ``y2`` are the displacements u[n] and u[n-1] with shape
    ``(batch, ndof, 1)``. The linear update is

        u[n+1] = T1 @ u[n] + T2 @ u[n-1].

    In the nonlinear modes the state-dependent local force f(u[n]) is
    subtracted through ``M_const``:

        u[n+1] = T1 @ u[n] + T2 @ u[n-1] - M_const @ f(u[n]).

    Fixed degrees of freedom ``fxp`` are held at zero. ``kmat`` is not used
    in the update.
    """
    if nonlinearity_mode is None:
        nonlinearity_mode = "unilateral" if use_nonlinear else "linear"
    if nonlinearity_mode not in VALID_NONLINEARITY_MODES:
        raise ValueError(
            f"Unsupported nonlinearity_mode={nonlinearity_mode!r}")

    # Hold the fixed (clamped) degrees of freedom at zero.
    h1 = y1.clone()
    h1[:, fxp, :] = y1[:, fxp, :] * 0
    h2 = y2.clone()
    h2[:, fxp, :] = y2[:, fxp, :] * 0

    # Cubic mode: bound the state so that delta**3 cannot overflow.
    if nonlinearity_mode == "cubic":
        h1 = torch.clamp(h1, min=-state_limit, max=state_limit)
        h2 = torch.clamp(h2, min=-state_limit, max=state_limit)
    
    # Linear part of the update.
    y_linear = torch.matmul(T1, h1) + torch.matmul(T2, h2)
    if nonlinearity_mode == "linear":
        return y_linear

    if nonlinearity_mode == "cubic":
        if k_cubic is None:
            raise ValueError("cubic mode requires k_cubic")
        nonlinear_force = build_cubic_force_local_resonators(
            h1, k_cubic)
        update_term = torch.einsum(
            "ij,bjk->bik", M_const, nonlinear_force)
        return torch.clamp(
            y_linear - update_term,
            min=-state_limit,
            max=state_limit)

    # Unilateral mode: the extra local spring k_d is active only while
    # u_main - u_local < 0.
    Knon = build_K_bilinear_delta_trainable(
        h1, k_d,
        switch_temperature=switch_temperature,
        use_surrogate_gradient=use_surrogate_gradient)
    v_temp = torch.bmm(Knon, h1)
    update_term = torch.einsum('ij, bjk -> bik', M_const, v_temp)

    return y_linear - update_term


def build_K_bilinear_delta(u, k_d):
    """Build the unilateral stiffness matrix for each batch item.

    The first half of ``u`` contains main-mass displacements and the second
    half contains their colocated local resonators. The additional spring at
    a cell is active only when ``u_main - u_local < 0``. Its 2x2 block is
    ``k_d * [[1, -1], [-1, 1]]``, which preserves action-reaction balance.

    Parameters
    ----------
    u:
        Displacement tensor with shape ``(batch, 2*n)`` or
        ``(batch, 2*n, 1)``.
    k_d:
        Unilateral stiffnesses with shape ``(1, n)`` or ``(n,)``.

    Returns
    -------
    torch.Tensor
        Batched stiffness matrices with shape ``(batch, 2*n, 2*n)``.
    """
    # Remove the trailing singleton field dimension used by WaveRNN.
    if u.dim() == 3 and u.shape[-1] == 1:
        u = u.squeeze(-1)
        
    B, N_total = u.shape 
    N_main = N_total // 2  # number of main masses (row * col)
    
    # Split the main-mass and local-resonator layers.
    u_M = u[:, :N_main]
    u_m = u[:, N_main:]
    
    # Preserve the displacement dtype when converting the Boolean switch.
    condition_mask = (u_M - u_m < 0).to(dtype=u.dtype)
    
    # Broadcasting applies the cell stiffnesses to every batch item.
    k_active = k_d * condition_mask
    
    # Assemble the independent 2x2 spring blocks without Python loops.
    K_delta = torch.zeros((B, N_total, N_total), device=u.device, dtype=u.dtype)
    
    idx_M = torch.arange(N_main, device=u.device)
    idx_m = idx_M + N_main
    
    # Off-diagonal terms couple each main mass to its local resonator.
    K_delta[:, idx_M, idx_m] = -k_active
    K_delta[:, idx_m, idx_M] = -k_active
    
    # Positive diagonal terms make each block symmetric and force-balanced.
    K_delta[:, idx_M, idx_M] = k_active
    K_delta[:, idx_m, idx_m] = k_active
    
    return K_delta


def build_K_bilinear_delta_trainable(
        u, k_d, switch_temperature=1.0,
        use_surrogate_gradient=False):
    """
    Build the same hard bilinear stiffness matrix as
    build_K_bilinear_delta, with an optional surrogate gradient.

    The forward value is always the physical switch
    (u_main - u_resonator < 0). When enabled, only the backward
    derivative near the switching boundary comes from a sigmoid.
    """
    if u.dim() == 3 and u.shape[-1] == 1:
        u = u.squeeze(-1)

    batch_size, n_total = u.shape
    n_main = n_total // 2
    relative_displacement = u[:, :n_main] - u[:, n_main:]
    hard_mask = (relative_displacement < 0).to(dtype=u.dtype)

    if use_surrogate_gradient and torch.is_grad_enabled():
        if switch_temperature <= 0:
            raise ValueError("switch_temperature must be positive")
        smooth_mask = torch.sigmoid(
            -relative_displacement / switch_temperature)
        condition_mask = (
            hard_mask + (smooth_mask - smooth_mask.detach()))
    else:
        condition_mask = hard_mask

    active_stiffness = k_d * condition_mask
    k_delta = torch.zeros(
        (batch_size, n_total, n_total),
        device=u.device, dtype=u.dtype)
    idx_main = torch.arange(n_main, device=u.device)
    idx_local = idx_main + n_main

    k_delta[:, idx_main, idx_main] = active_stiffness
    k_delta[:, idx_local, idx_local] = active_stiffness
    k_delta[:, idx_main, idx_local] = -active_stiffness
    k_delta[:, idx_local, idx_main] = -active_stiffness
    return k_delta


class WaveCell(torch.nn.Module):
    """One recurrent time step of the resonator lattice.

    The cell owns the ``Coupling`` module (physical parameters) and the
    nonlinearity settings. The time-invariant operators T1, T2 and M_const
    are assembled once per sequence by ``WaveRNN`` and passed in.
    """


    def __init__(self,
                 dt : float,
                 coupling,
                 switch_temperature: float = 1.0,
                 use_surrogate_gradient: bool = False,
                 use_nonlinear: bool = True,
                 nonlinearity_mode=None,
                 state_limit: float = 5000.0):

        super().__init__()

        # Set values
        self.register_buffer("dt", to_tensor(dt))
        self.coup = coupling
        if switch_temperature <= 0:
            raise ValueError("switch_temperature must be positive")
        if state_limit <= 0:
            raise ValueError("state_limit must be positive")
        if nonlinearity_mode is None:
            nonlinearity_mode = (
                "unilateral" if use_nonlinear else "linear")
        if nonlinearity_mode not in VALID_NONLINEARITY_MODES:
            raise ValueError(
                "nonlinearity_mode must be one of "
                f"{sorted(VALID_NONLINEARITY_MODES)}")
        self.switch_temperature = float(switch_temperature)
        self.use_surrogate_gradient = bool(use_surrogate_gradient)
        self.nonlinearity_mode = nonlinearity_mode
        self.use_nonlinear = nonlinearity_mode != "linear"
        self.state_limit = float(state_limit)

    def forward(
            self, T1, T2, M_const, kmat, h1, h2, fxp, k_d,
            k_cubic=None):
        """Return ``(u[n+1], u[n])`` for the next recurrence step."""

#         y = _time_step(k_n, k_c, m_n, m_c, c_n, c_c, h1, h2, self.dt, coupMat, fxp)
        y = _time_step(
            T1, T2, M_const, kmat, h1, h2, fxp, k_d,
            switch_temperature=getattr(
                self, "switch_temperature", 1.0),
            use_surrogate_gradient=getattr(
                self, "use_surrogate_gradient", False),
            use_nonlinear=getattr(
                self, "use_nonlinear", True),
            nonlinearity_mode=getattr(
                self, "nonlinearity_mode", None),
            k_cubic=(
                self.coup.k_cubic if k_cubic is None else k_cubic),
            state_limit=getattr(self, "state_limit", 5000.0))

        return y, h1
