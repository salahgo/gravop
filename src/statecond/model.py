"""statecond.model — one operator architecture, four domains.

`OperatorNet` is `gravop.model.GravOpNet` with the SIDM-specific pieces removed:
the number of input/output fields, the number of physical parameters and the
number of conditioning variables are arguments rather than constants. The
spectral backbone and the FiLM conditioning are imported from `gravop.model`
unchanged, so a difference between the tau- and state-conditioned runs cannot
come from a difference in the network.

`TimeHead` predicts log10 t_sing from the initial state alone. It is what a
deployed tau-conditioned surrogate needs and does not have: to answer "what
does this system look like at physical time t?" the user must first convert t
to tau = t / t_sing, and t_sing is unknown until the singularity happens.
"""
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gravop.model import SpectralConv1d, FiLM  # noqa: E402


class OperatorNet(nn.Module):
    """FNO with FiLM conditioning, agnostic to the physical system.

    prof: (B, C, N) initial fields, normalised
    cond: (B, cond_dim) = physical parameters followed by the conditioning
          variable(s) -- either (tau,) or (phi, flag)
    returns fields (B, C, N)
    """

    def __init__(self, n_fields=1, n_par=4, cond_extra=1, n_x=128, width=64,
                 modes=32, n_layers=4, cond_embed=64):
        super().__init__()
        cond_dim = n_par + cond_extra
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_embed), nn.GELU(),
            nn.Linear(cond_embed, cond_embed), nn.GELU())
        self.lift = nn.Conv1d(n_fields + 1, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.film = nn.ModuleList(
            [FiLM(cond_embed, width) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv1d(width, 128, 1), nn.GELU(), nn.Conv1d(128, n_fields, 1))
        self.register_buffer('coord', torch.linspace(-1.0, 1.0, n_x))

    def forward(self, prof, cond):
        B, _, N = prof.shape
        c = (self.coord if N == self.coord.shape[0]
             else torch.linspace(-1.0, 1.0, N, device=prof.device,
                                 dtype=prof.dtype))
        x = self.lift(torch.cat([prof, c[None, None, :].expand(B, 1, N)], 1))
        ce = self.cond_mlp(cond)
        for sp, pw, fl in zip(self.spectral, self.pointwise, self.film):
            x = fl(F.gelu(sp(x) + pw(x)), ce) + x
        return self.proj(x)


class DeepONetOperator(nn.Module):
    """Same interface as `OperatorNet`, different architecture.

    The conditioning claim is about *which variable* the operator is told, not
    about the network that consumes it, but so far it has only been shown inside
    one backbone. This is the standard operator-learning alternative, in its
    usual form: a branch net on the initial field and the physical parameters, a
    trunk net on the query coordinates — here the spatial coordinate together
    with whichever conditioning variable is being tested — and an inner product
    between them.
    """

    def __init__(self, n_fields=1, n_par=4, cond_extra=1, n_x=128, p=128,
                 width=256):
        super().__init__()
        self.n_fields, self.p = n_fields, p
        self.branch = nn.Sequential(
            nn.Linear(n_fields * n_x + n_par, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, n_fields * p))
        self.trunk = nn.Sequential(
            nn.Linear(1 + cond_extra, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, p))
        self.n_par = n_par
        self.register_buffer('coord', torch.linspace(-1.0, 1.0, n_x))

    def forward(self, prof, cond):
        B, _, N = prof.shape
        c = (self.coord if N == self.coord.shape[0]
             else torch.linspace(-1.0, 1.0, N, device=prof.device,
                                 dtype=prof.dtype))
        par, extra = cond[:, :self.n_par], cond[:, self.n_par:]
        b = self.branch(torch.cat([prof.reshape(B, -1), par], 1))
        b = b.reshape(B, self.n_fields, self.p)
        # trunk sees (x, conditioning) at every grid point
        tr_in = torch.cat([
            c[None, :, None].expand(B, N, 1),
            extra[:, None, :].expand(B, N, extra.shape[1])], dim=-1)
        t = self.trunk(tr_in)                                  # (B, N, p)
        # The natural contraction here is einsum('bcp,bnp->bcn'). With
        # n_fields = 1 that is a batched outer product, and its BACKWARD is
        # routed to a Triton kernel that this machine cannot JIT-compile
        # (no Python development headers, so cuda_utils.c fails to build).
        # The broadcast form is mathematically identical, uses only elementwise
        # ops and a reduction, and costs B*C*N*p floats -- 67 MB at the batch
        # size used here, which is acceptable. See STATE.md, environment notes.
        return (b.unsqueeze(2) * t.unsqueeze(1)).sum(-1) / np.sqrt(self.p)


def band_limited_resample(x, n_out):
    """Resample a periodic 1-D signal to `n_out` points without aliasing.

    Truncating or zero-padding the rfft is the ideal (sinc) low-pass filter,
    which on a periodic grid is exact rather than the Lanczos approximation the
    CNO paper uses on bounded domains. `rfft` is unnormalised and `irfft`
    divides by its output length, so the amplitude is restored by n_out / n_in.
    """
    n_in = x.shape[-1]
    if n_in == n_out:
        return x
    xf = torch.fft.rfft(x)
    n_keep = min(xf.shape[-1], n_out // 2 + 1)
    out = torch.zeros(*x.shape[:-1], n_out // 2 + 1,
                      dtype=xf.dtype, device=x.device)
    out[..., :n_keep] = xf[..., :n_keep]
    return torch.fft.irfft(out, n=n_out) * (n_out / n_in)


class BandLimitedActivation(nn.Module):
    """CNO's activation: act on a finer grid, then filter back down.

    A pointwise nonlinearity applied to a band-limited function is not itself
    band-limited — it produces frequencies above the grid's Nyquist that then
    alias. Raonic et al. (2023) make the block a genuine operator between
    band-limited spaces by upsampling, activating, and low-pass filtering back.
    This is the one thing that distinguishes CNO from an ordinary 1-D U-Net.
    """

    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        n = x.shape[-1]
        return band_limited_resample(
            F.gelu(band_limited_resample(x, n * self.factor)), n)


def _groups_for(ch, want=8):
    """Largest divisor of `ch` not exceeding `want`, so GroupNorm always builds
    whatever channel widths the level scheme produces."""
    return max(g for g in range(1, min(want, ch) + 1) if ch % g == 0)


class _CNOBlock(nn.Module):
    """Resample, convolve, condition, activate."""

    def __init__(self, c_in, c_out, cond_embed, n_out=None, groups=8):
        super().__init__()
        self.n_out = n_out
        self.conv = nn.Conv1d(c_in, c_out, 3, padding=1,
                              padding_mode='circular')
        self.norm = nn.GroupNorm(_groups_for(c_out, groups), c_out)
        self.film = FiLM(cond_embed, c_out)
        self.act = BandLimitedActivation()

    def forward(self, x, ce):
        if self.n_out is not None:
            x = band_limited_resample(x, self.n_out)
        return self.act(self.film(self.norm(self.conv(x)), ce))


class _CNOResBlock(nn.Module):
    """Resolution- and channel-preserving residual block."""

    def __init__(self, ch, cond_embed, groups=8):
        super().__init__()
        self.b1 = _CNOBlock(ch, ch, cond_embed, groups=groups)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1, padding_mode='circular')
        self.norm2 = nn.GroupNorm(_groups_for(ch, groups), ch)
        self.film2 = FiLM(cond_embed, ch)

    def forward(self, x, ce):
        return x + self.film2(self.norm2(self.conv2(self.b1(x, ce))), ce)


class CNOOperator(nn.Module):
    """Convolutional neural operator, same interface as `OperatorNet`.

    A U-shaped encoder/decoder in which every resolution change and every
    nonlinearity is band-limited, so the network approximates an operator
    between function spaces rather than between grids. It is the architecture
    reported to beat FNO on the standard operator-learning benchmarks, and it is
    a different *family* from both FNO (global spectral) and DeepONet (branch /
    trunk inner product) — which is what E27 found the ranking depends on.

    Conditioning enters exactly as in `OperatorNet`: one FiLM layer per block
    driven by the same embedding MLP, so a difference between the tau- and
    state-conditioned arms still cannot come from the network.
    """

    # width 32 / n_res 3 is chosen to land at 577k parameters against the FNO's
    # 587k, so "stronger backbone" cannot be read as "bigger backbone"
    def __init__(self, n_fields=1, n_par=4, cond_extra=1, n_x=128, width=32,
                 levels=2, n_res=3, cond_embed=64):
        super().__init__()
        cond_dim = n_par + cond_extra
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_embed), nn.GELU(),
            nn.Linear(cond_embed, cond_embed), nn.GELU())
        self.lift = nn.Conv1d(n_fields + 1, width, 1)

        chs = [width * 2 ** i for i in range(levels + 1)]
        self.down = nn.ModuleList()
        for i in range(levels):
            self.down.append(_CNOBlock(chs[i], chs[i + 1], cond_embed,
                                       n_out=None))   # resolution set at call
        self.mid = nn.ModuleList(
            [_CNOResBlock(chs[-1], cond_embed) for _ in range(n_res)])
        # CNO passes each skip through an invariant block rather than
        # concatenating the raw encoder feature
        self.skip = nn.ModuleList(
            [_CNOResBlock(chs[i], cond_embed) for i in range(levels)])
        self.up = nn.ModuleList()
        for i in reversed(range(levels)):
            self.up.append(_CNOBlock(chs[i + 1] + chs[i], chs[i], cond_embed))
        self.proj = nn.Sequential(
            nn.Conv1d(width, 128, 1), nn.GELU(), nn.Conv1d(128, n_fields, 1))
        self.levels = levels
        self.register_buffer('coord', torch.linspace(-1.0, 1.0, n_x))

    def forward(self, prof, cond):
        B, _, N = prof.shape
        c = (self.coord if N == self.coord.shape[0]
             else torch.linspace(-1.0, 1.0, N, device=prof.device,
                                 dtype=prof.dtype))
        x = self.lift(torch.cat([prof, c[None, None, :].expand(B, 1, N)], 1))
        ce = self.cond_mlp(cond)

        skips = []
        for i, blk in enumerate(self.down):
            skips.append(self.skip[i](x, ce))
            n_half = max(4, x.shape[-1] // 2)
            x = band_limited_resample(x, n_half)
            x = blk(x, ce)
        for blk in self.mid:
            x = blk(x, ce)
        for blk, s in zip(self.up, reversed(skips)):
            x = band_limited_resample(x, s.shape[-1])
            x = blk(torch.cat([x, s], 1), ce)
        return self.proj(x)


class _AttnBlock(nn.Module):
    """Pre-norm self-attention + feed-forward, both FiLM-conditioned."""

    def __init__(self, width, n_heads, cond_embed, mlp_ratio=2):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.film1 = FiLM(cond_embed, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, mlp_ratio * width), nn.GELU(),
            nn.Linear(mlp_ratio * width, width))
        self.film2 = FiLM(cond_embed, width)

    def _film(self, film, t, ce):
        # FiLM is written for (B, C, N); tokens here are (B, N, C)
        return film(t.transpose(1, 2), ce).transpose(1, 2)

    def forward(self, t, ce):                       # t: (B, N, C)
        B, N, C = t.shape
        h = self.norm1(t)
        q, k, v = self.qkv(h).reshape(B, N, 3, self.n_heads,
                                      C // self.n_heads).unbind(2)
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        a = self.out(a.transpose(1, 2).reshape(B, N, C))
        t = t + self._film(self.film1, a, ce)
        return t + self._film(self.film2, self.mlp(self.norm2(t)), ce)


class TransformerOperator(nn.Module):
    """Attention-based operator (OFormer / Transolver family), same interface.

    Every grid point is a token carrying the local field value and a Fourier
    positional encoding of its coordinate; blocks of global self-attention then
    mix them. Unlike FNO the mixing is *input-dependent* rather than a fixed set
    of spectral multipliers, which is the property these architectures are
    argued to gain from. With 128 points, full attention costs nothing.
    """

    # width 120 gives 616k parameters against the FNO's 587k — again matched, so
    # any accuracy difference is attributable to the architecture, not capacity
    def __init__(self, n_fields=1, n_par=4, cond_extra=1, n_x=128, width=120,
                 n_layers=4, n_heads=8, n_freq=16, cond_embed=64):
        super().__init__()
        cond_dim = n_par + cond_extra
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_embed), nn.GELU(),
            nn.Linear(cond_embed, cond_embed), nn.GELU())
        self.n_freq = n_freq
        self.lift = nn.Linear(n_fields + 1 + 2 * n_freq, width)
        self.blocks = nn.ModuleList(
            [_AttnBlock(width, n_heads, cond_embed) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, n_fields))
        self.register_buffer('coord', torch.linspace(-1.0, 1.0, n_x))
        self.register_buffer(
            'freqs', 2.0 ** torch.arange(n_freq).float() * np.pi)

    def forward(self, prof, cond):
        B, _, N = prof.shape
        c = (self.coord if N == self.coord.shape[0]
             else torch.linspace(-1.0, 1.0, N, device=prof.device,
                                 dtype=prof.dtype))
        ang = c[:, None] * self.freqs[None, :]                  # (N, n_freq)
        pe = torch.cat([ang.sin(), ang.cos()], -1)              # (N, 2*n_freq)
        t = torch.cat([prof.transpose(1, 2),
                       c[None, :, None].expand(B, N, 1),
                       pe[None].expand(B, N, 2 * self.n_freq)], -1)
        t = self.lift(t)
        ce = self.cond_mlp(cond)
        for blk in self.blocks:
            t = blk(t, ce)
        return self.proj(self.norm(t)).transpose(1, 2)


BACKBONES = {'fno': OperatorNet, 'deeponet': DeepONetOperator,
             'cno': CNOOperator, 'transformer': TransformerOperator}


class TimeHead(nn.Module):
    """Predict log10 t_sing from the initial fields and the parameters."""

    def __init__(self, n_fields=1, n_par=4, n_x=128, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_fields * n_x + n_par, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, prof, par):
        return self.net(torch.cat([prof.reshape(prof.shape[0], -1), par], 1))[:, 0]


class PhaseHead(nn.Module):
    """Predict the state coordinate (phi, flag) from a *noisy observation* of
    the current field.

    Provided as an extension point; NOT used by any reported experiment. The
    docstring previously claimed it was "the estimator used in the noise
    study", which was false: E21/E23 perturb phi directly (phi + N(0, eps),
    branch flipped with probability q) rather than estimating it, and P13
    trains its own classifier for the branch (`scripts/sc_contract.py`). Kept
    because the benchmark guide offers it to third parties, corrected because
    a claim of use is a claim about results.
    """

    def __init__(self, n_fields=1, n_x=128, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_fields * n_x, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, prof_now):
        out = self.net(prof_now.reshape(prof_now.shape[0], -1))
        return out[:, 0], out[:, 1]      # phi_hat, flag_logit
