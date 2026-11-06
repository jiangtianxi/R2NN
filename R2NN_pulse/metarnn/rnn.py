import torch
from . import construct

class WaveRNN(torch.nn.Module):
    """Second-order recurrent solver for the mechanical lattice.

    Parameters
    ----------
    cell : WaveCell
        One time step of the lattice dynamics.
    sources : WaveSource or list of WaveSource
        Input injection point(s).
    dt : float
        Time step.
    fxp : sequence of int
        Fixed (clamped) degrees of freedom.
    srcpos : int
        Degree of freedom that receives the input.
    probes : WaveProbe or list of WaveProbe
        Output readouts; one probe per class.
    """

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
        if_test :
            Not used.

        Returns
        -------
        torch.Tensor
            Probe outputs with shape ``(batch, time, n_probes)``, or the
            displacement field with shape ``(batch, time, ndof, 1)``
            when ``output_fields`` is True or no probe is defined.
        """

        parameter = next(self.parameters())
        device = parameter.device
        if x.device != device:
            raise ValueError(
                f"Input is on {x.device}, but the model is on {device}")
        if x.dtype != parameter.dtype:
            raise ValueError(
                f"Input dtype is {x.dtype}, but the model dtype is "
                f"{parameter.dtype}")

        # First dim is batch
        batch_size = x.shape[0]

        # Init hidden states
        hidden_state_shape = (batch_size,) + self.cell.coup.domain_shape
        # h1 = torch.zeros(hidden_state_shape, dtype=torch.float32)
        # h2 = torch.zeros(hidden_state_shape, dtype=torch.float32)
#         h1 = torch.zeros(hidden_state_shape, dtype=torch.float32, device=device)
#         h2 = torch.zeros(hidden_state_shape, dtype=torch.float32, device=device)
        h1 = torch.zeros(hidden_state_shape, dtype=x.dtype, device=device)
        h2 = torch.zeros(hidden_state_shape, dtype=x.dtype, device=device)
        y_all = []

        # These quantities do not change with time. Pulling them out of the loop avoids
        # recomputing them at every time step and reduces the memory used by backpropagation.

        k_n_param = self.cell.coup.k_n
        k_c_param = self.cell.coup.k_c
        k_d_param = self.cell.coup.k_d
        k_cubic = self.cell.coup.k_cubic
        m_n = self.cell.coup.m_n
        m_c = self.cell.coup.m_c
        c_n = self.cell.coup.c_n
        c_c = self.cell.coup.c_c
        c_g = self.cell.coup.c_g
        coupMat = self.cell.coup.coupMat
        

        # The optimizer projection removes linear springs below its configured
        # threshold. Exact zeros are preserved during training and inference;
        # the clamp only guards against negative values.
#         k_n = torch.clamp(k_n_param, min=1e-10, max=0.06)
#         k_c = torch.clamp(k_c_param, min=1e-10, max=0.04)
        k_n = torch.clamp(k_n_param, min=0.0)
        k_c = torch.clamp(k_c_param, min=0.0)
        nonlinearity_mode = getattr(
            self.cell,
            "nonlinearity_mode",
            "unilateral" if getattr(
                self.cell, "use_nonlinear", True) else "linear")
        if nonlinearity_mode == "unilateral":
            k_d = torch.clamp(k_d_param, min=1e-8)
        else:
            # Linear and cubic modes bypass the unilateral spring.
            k_d = torch.zeros_like(k_d_param)

        # Assemble the linear physical operators once per input sequence.
# 		if if_test==False:
# 			k_n = self.cell.coup.k_n
# 			k_c = self.cell.coup.k_c     
# 			m_n = self.cell.coup.m_n
# 			m_c = self.cell.coup.m_c
# 			c_n = self.cell.coup.c_n
# 			c_c = self.cell.coup.c_c
# 			coupMat = self.cell.coup.coupMat

# 			mean_val = 0e-3*torch.ones(k_c.size())
# 			k_c = torch.where(k_c > 0, k_c, mean_val)  

# 			mean_val = 0e-3*torch.ones(k_n.size())
# 			k_n = torch.where(k_n > 0, k_n, mean_val) 

# 			self.cell.coup.kc = k_c
# 			self.cell.coup.kn = k_n
# 		else:
# 			k_n = self.cell.coup.kn
# 			k_c = self.cell.coup.kc     
# 			m_n = self.cell.coup.m_n
# 			m_c = self.cell.coup.m_c
# 			c_n = self.cell.coup.c_n
# 			c_c = self.cell.coup.c_c
# 			coupMat = self.cell.coup.coupMat

        # Global stiffness, mass and damping matrices.
        kmat, mmat, cmat = construct.construct_MKC(
            k_n, k_c, m_n, m_c, c_n, c_c, coupMat, c_g)

        num, _ = kmat.shape
        fixed_idx = torch.as_tensor(
            self.fxp, device=device, dtype=torch.long).reshape(-1)
        if fixed_idx.numel():
            fixed_idx = torch.unique(fixed_idx)
            if fixed_idx.min() < 0 or fixed_idx.max() >= num:
                raise IndexError(
                    f"Fixed-node indices must be in [0, {num - 1}]")

        # Free (unclamped) degrees of freedom.
        free_mask = torch.ones(num, device=device, dtype=torch.bool)
        free_mask[fixed_idx] = False
        free_idx = torch.arange(num, device=device)[free_mask]
        if free_idx.numel() == 0:
            raise ValueError("At least one free degree of freedom is required")

        # The source must not sit on a fixed node.
        if len(self.sources):
            source_positions = torch.stack(
                [source.x.reshape(()) for source in self.sources])
            source_is_fixed = (
                source_positions[:, None] == fixed_idx[None, :]).any()
            if source_is_fixed:
                raise ValueError("A source cannot be placed on a fixed node")
        if self.srcpos < 0 or self.srcpos >= num:
            raise IndexError(f"srcpos must be in [0, {num - 1}]")
        if fixed_idx.numel() and (fixed_idx == self.srcpos).any():
            raise ValueError("srcpos cannot be a fixed node")

        # Apply the Dirichlet constraint before matrix inversion.
        kmat_free = kmat.index_select(
            0, free_idx).index_select(1, free_idx)
        mmat_free = mmat.index_select(
            0, free_idx).index_select(1, free_idx)
        cmat_free = cmat.index_select(
            0, free_idx).index_select(1, free_idx)
        phi_free = torch.linalg.inv(mmat_free)

        # Precompute the time-invariant central-difference operators. The
        # fixed degrees of freedom are eliminated before matrix inversion so
        # the supports impose exact Dirichlet constraints.
        num_free = free_idx.numel()
        I_free = torch.eye(
            num_free, device=device, dtype=phi_free.dtype)
        # Shared implicit damping factor:
        # (I + dt/2 * M^-1 C)^-1.
        inv_term_free = torch.linalg.inv(
            I_free
            + (self.dt / 2) * torch.matmul(phi_free, cmat_free))

        # Maps a generalized nodal force to the displacement update.
        deno_free = torch.matmul(inv_term_free, phi_free)

        # State transition for u[n+1] = T1*u[n] + T2*u[n-1] + ...
        T1_free = torch.matmul(
            inv_term_free,
            2 * I_free
            - (self.dt**2) * torch.matmul(phi_free, kmat_free))
        T2_free = torch.matmul(
            inv_term_free,
            -I_free
            + (self.dt / 2) * torch.matmul(phi_free, cmat_free))

        # Multiplier for the explicit state-dependent stiffness force.
        M_const_free = (self.dt ** 2) * deno_free

        # Lift reduced operators back to the original coordinate system so
        # existing source/probe indices remain valid. Fixed rows and columns
        # are exactly zero.
        projection = kmat.new_zeros((num, num_free))
        projection[free_idx, torch.arange(num_free, device=device)] = 1
        projection_t = projection.transpose(0, 1)
        T1 = torch.matmul(torch.matmul(
            projection, T1_free), projection_t)
        T2 = torch.matmul(torch.matmul(
            projection, T2_free), projection_t)
        M_const = torch.matmul(torch.matmul(
            projection, M_const_free), projection_t)
        deno = torch.matmul(torch.matmul(
            projection, deno_free), projection_t)
        # -------------------------------------------------------------




# 		deno = torch.matmul(torch.linalg.inv(torch.eye(num) + self.dt / 2 * torch.matmul(phi, cmat)), phi)

# 		h2[:, self.srcpos, :] = self.dt**2 * x[:, self.srcpos] / 2 / mmat[self.srcpos, self.srcpos].numpy()
        # Seed the previous-step state at the source with the first input
        # sample.
        m_val = mmat[self.srcpos, self.srcpos]
        h2[:, self.srcpos, :] = (
            self.dt**2 * x[:, 0] / 2 / m_val)

        # Loop through time
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):

            # Propagate the fields
# 			h1, h2 = self.cell(k_n, k_c, m_n, m_c, c_n, c_c, h1, h2, coupMat, self.fxp)
            h1, h2 = self.cell(
                T1, T2, M_const, kmat, h1, h2,
                self.fxp, k_d, k_cubic)

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
