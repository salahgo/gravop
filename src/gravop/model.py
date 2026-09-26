"""gravop.model — neural operator emulator for gravothermal evolution.

GravOpNet: input = initial profile functions (log10 rho_0, log10 v_0) on the fixed
log-r grid + conditioning vector c = (log10 sigma0, log10 w, model_id, tau).
Output = (log10 rho, log10 v) at time tau. A scalar head predicts log10 t_c and
a no-collapse logit from the profile + parameters (tau-independent).

Backbone: 1D Fourier layers (spectral convolution over the log-r axis) with
FiLM conditioning after each layer. Physics residuals are computed in
physical (dimensionless solver) units by undoing the data normalization.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, ch_in, ch_out, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (ch_in * ch_out)
        self.w = nn.Parameter(
            scale * torch.randn(ch_in, ch_out, modes, dtype=torch.cfloat))

    def forward(self, x):  # x: (B, C, N)
        xf = torch.fft.rfft(x)
        out = torch.zeros(x.size(0), self.w.size(1), xf.size(-1),
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :self.modes] = torch.einsum(
            'bim,iom->bom', xf[:, :, :self.modes], self.w)
        return torch.fft.irfft(out, n=x.size(-1))


class FiLM(nn.Module):
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * ch)

    def forward(self, x, c):  # x: (B, C, N), c: (B, cond_dim)
        gb = self.lin(c)[:, :, None]
        g, b = gb.chunk(2, dim=1)
        return x * (1 + g) + b


class GravOpNet(nn.Module):
    def __init__(self, n_r=128, width=64, modes=32, n_layers=4, cond_dim=4,
                 cond_embed=64):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_embed), nn.GELU(),
            nn.Linear(cond_embed, cond_embed), nn.GELU())
        self.lift = nn.Conv1d(3, width, 1)  # (lrho0, lv0, log r)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.film = nn.ModuleList(
            [FiLM(cond_embed, width) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv1d(width, 128, 1), nn.GELU(), nn.Conv1d(128, 2, 1))
        # scalar head: pooled lifted features + params (not tau)
        self.head = nn.Sequential(
            nn.Linear(width + 3, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2))

        lr = torch.linspace(-2.5, 2.0, n_r)
        self.register_buffer('log_r', lr)

    def forward(self, prof, cond):
        """prof: (B, 2, N) initial profiles (normalized);
        cond: (B, 4) = (lsig, lw, model_id, tau) (lsig/lw normalized).
        Returns fields (B, 2, N), scalars (B, 2) = (log_tc_norm, nc_logit).

        N may differ from the training resolution: the spectral layers act on
        a fixed set of Fourier modes and the coordinate channel is rebuilt for
        the incoming grid, so the operator can be evaluated zero-shot on a
        finer (or coarser) radial discretization of the same log-r interval."""
        B, _, N = prof.shape
        if N == self.log_r.shape[0]:
            lr = self.log_r
        else:
            lr = torch.linspace(float(self.log_r[0]), float(self.log_r[-1]), N,
                                device=prof.device, dtype=prof.dtype)
        grid = lr[None, None, :].expand(B, 1, N) / 2.5
        x = self.lift(torch.cat([prof, grid], dim=1))
        pooled = x.mean(dim=-1)
        ce = self.cond_mlp(cond)
        for sp, pw, fl in zip(self.spectral, self.pointwise, self.film):
            x = fl(F.gelu(sp(x) + pw(x)), ce) + x
        fields = self.proj(x)
        scalars = self.head(torch.cat([pooled, cond[:, :3]], dim=1))
        return fields, scalars


class DeepONet(nn.Module):
    """Baseline: branch on (profiles + params), trunk on (log r, tau)."""
    def __init__(self, n_r=128, p=128, width=256):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(2 * n_r + 3, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2 * p))
        self.trunk = nn.Sequential(
            nn.Linear(2, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, p))
        self.p = p
        self.head = nn.Sequential(
            nn.Linear(2 * n_r + 3, 128), nn.GELU(), nn.Linear(128, 2))
        lr = torch.linspace(-2.5, 2.0, n_r)
        self.register_buffer('log_r', lr)

    def forward(self, prof, cond):
        B, _, N = prof.shape
        flat = torch.cat([prof.reshape(B, -1), cond[:, :3]], dim=1)
        b = self.branch(flat).reshape(B, 2, self.p)
        tr_in = torch.stack([
            self.log_r[None, :].expand(B, N) / 2.5,
            cond[:, 3:4].expand(B, N)], dim=-1)
        t = self.trunk(tr_in)  # (B, N, p)
        fields = torch.einsum('bcp,bnp->bcn', b, t) / np.sqrt(self.p)
        scalars = self.head(flat)
        return fields, scalars


# ----------------------------------------------------------------------------
# physics residuals (differentiable, act on de-normalized log10 fields)

LN10 = float(np.log(10.0))


def phys_residuals(lrho, lv, log_r):
    """lrho, lv: (B, N) log10 fields on grid log_r (log10). Returns
    (hyd, mass, energy) diagnostics:
      hyd: normalized hydrostatic residual (B, N-2)
      mtot: total mass within grid (B,)
      etot: total energy within grid (B,)
    Mass/energy conservation losses compare mtot/etot across tau outside."""
    r = 10.0 ** log_r  # (N,)
    # clamp to the physically populated range so early-training garbage cannot
    # produce astronomic integrals (rho in [1e-9, 1e7], v in [1e-4, 30])
    rho = 10.0 ** lrho.clamp(-9.0, 7.0)
    v = 10.0 ** lv.clamp(-4.0, 1.5)
    p = rho * v * v
    dlnr = (log_r[1] - log_r[0]) * LN10
    # cumulative mass: dm = r^3 rho dlnr (units absorb 4pi)
    integrand = (r ** 3) * rho
    m = torch.cumsum(0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dlnr, dim=1)
    m = F.pad(m, (1, 0))
    # hydrostatic: dp/dlnr = - m rho / r
    dp = (p[:, 2:] - p[:, :-2]) / (2 * dlnr)
    rhs = -m[:, 1:-1] * rho[:, 1:-1] / r[1:-1]
    scale = p[:, 1:-1] + (m[:, 1:-1] * rho[:, 1:-1] / r[1:-1]).abs()
    hyd = (dp - rhs) / (scale + 1e-12)
    mtot = m[:, -1]
    # energy: U = 1.5 p per volume -> dU = 1.5 p r^3 dlnr ; W = -m dm / r
    uint = 1.5 * p * r ** 3
    U = torch.sum(0.5 * (uint[:, 1:] + uint[:, :-1]) * dlnr, dim=1)
    dm = m[:, 1:] - m[:, :-1]
    rmid = 0.5 * (r[1:] + r[:-1])
    mmid = 0.5 * (m[:, 1:] + m[:, :-1])
    W = -torch.sum(mmid * dm / rmid, dim=1)
    return hyd, mtot, U + W
