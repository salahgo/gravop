"""Physical-time forecasting with a learned clock and local/spectral residuals.

The forward interface never accepts target phase, branch, or termination time.
The clock is supervised separately and frozen while the field model is fitted.
All normalizations, including the clock output scale, travel in the checkpoint.
"""
import torch
from torch import nn
from torch.nn import functional as F

from gravop.model import FiLM
from .model import TimeHead


class SpectralLocalBlock(nn.Module):
    def __init__(self, width, modes, cond_width):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, width, modes,
                                               dtype=torch.cfloat) / width)
        self.local = nn.Conv1d(width, width, 5, padding=2)
        self.point = nn.Conv1d(width, width, 1)
        self.film = FiLM(cond_width, width)

    def forward(self, x, cond):
        xf = torch.fft.rfft(x)
        m = min(xf.shape[-1], self.weight.shape[-1])
        yf = torch.zeros_like(xf)
        # Explicit reduction also supports singleton fields on the local CUDA
        # build without routing backward through an unavailable Triton compiler.
        yf[..., :m] = torch.einsum('bim,iom->bom', xf[..., :m],
                                   self.weight[..., :m])
        y = torch.fft.irfft(yf, n=x.shape[-1])
        return x + 0.5 * self.film(F.gelu(y + self.local(x) + self.point(x)), cond)


class ClockConditionedOperator(nn.Module):
    """Forecast normalized fields from normalized initial fields/parameters.

    ``time`` is physical time in the dataset's units, shape (batch,).
    ``clock_mu/sd`` invert the training-only normalization of log10(t_end).
    ``time_scale`` is a training-only physical time scale. Validity beyond the
    learned terminal event is exposed by ``clock_ratio``, not silently asserted.
    The clock encoder expects the training spatial resolution; field resolution
    transfer is therefore intentionally outside this model's public contract.
    """
    def __init__(self, n_fields, n_par, n_x=128, width=64, modes=32,
                 n_layers=4, clock_mu=0., clock_sd=1., time_scale=1.,
                 local=True, aligned=True):
        super().__init__()
        self.n_x, self.aligned = n_x, aligned
        self.clock = TimeHead(n_fields=n_fields, n_par=n_par, n_x=n_x)
        self.clock.requires_grad_(False)
        self.register_buffer('clock_mu', torch.tensor(float(clock_mu)))
        self.register_buffer('clock_sd', torch.tensor(float(clock_sd)))
        self.register_buffer('time_scale', torch.tensor(float(time_scale)))
        self.register_buffer('coord', torch.linspace(-1., 1., n_x))
        extra = 4 if aligned else 1
        self.condition = nn.Sequential(nn.Linear(n_par + extra, width), nn.GELU(),
                                        nn.Linear(width, width), nn.GELU())
        self.lift = nn.Conv1d(n_fields + 1, width, 1)
        self.blocks = nn.ModuleList([SpectralLocalBlock(width, modes, width)
                                     for _ in range(n_layers)])
        if not local:
            for block in self.blocks:
                nn.init.zeros_(block.local.weight)
                nn.init.zeros_(block.local.bias)
                block.local.requires_grad_(False)
        self.project = nn.Sequential(nn.Conv1d(width, 128, 1), nn.GELU(),
                                      nn.Conv1d(128, n_fields, 1))

    def clock_ratio(self, prof, par, time):
        log_end = self.clock(prof, par) * self.clock_sd + self.clock_mu
        return time / torch.pow(10., log_end)

    def forward(self, prof, par, time):
        if prof.shape[-1] != self.n_x:
            raise ValueError(f'clock encoder requires {self.n_x} spatial points')
        lt = torch.log1p(time / self.time_scale)
        if self.aligned:
            q = self.clock_ratio(prof, par, time)
            qc = q.clamp(0., 1.2)
            # Softened distance resolves the terminal boundary while remaining
            # finite for a clock that predicts termination slightly too early.
            near = -torch.log((1. - qc).clamp_min(1e-3))
            c = torch.cat([par, lt[:, None], qc[:, None], near[:, None],
                           torch.asinh(10. * (qc - .9))[:, None]], dim=1)
            gate = -torch.expm1(-10. * qc)
        else:
            c = torch.cat([par, lt[:, None]], dim=1)
            gate = -torch.expm1(-10. * time / self.time_scale)
        ce = self.condition(c)
        grid = self.coord[None, None].expand(prof.shape[0], 1, -1)
        x = self.lift(torch.cat([prof, grid], dim=1))
        for block in self.blocks:
            x = block(x, ce)
        return prof + gate[:, None, None] * self.project(x)


class ForecastPredictor:
    """Load a trusted local checkpoint and predict in dataset field units.

    Field units are log10 fields for the positive systems and the stored signed
    field for Burgers. Parameters follow the dataset's ``param_names`` ordering.
    No dataset file or training script is needed at inference.
    """
    def __init__(self, checkpoint, device='cpu'):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        self.device = device
        self.model = ClockConditionedOperator(**ck['config']['model']).to(device)
        self.model.load_state_dict(ck['state'])
        self.model.eval()
        n = ck['norm']
        self.f_mu = torch.as_tensor(n['f_mu'], device=device).float().reshape(1,-1,1)
        self.f_sd = torch.as_tensor(n['f_sd'], device=device).float().reshape(1,-1,1)
        self.p_mu = torch.as_tensor(n['p_mu'], device=device).float().reshape(1,-1)
        self.p_sd = torch.as_tensor(n['p_sd'], device=device).float().reshape(1,-1)
        scales = (self.f_sd, self.p_sd, self.model.clock_sd, self.model.time_scale)
        if (not all(torch.isfinite(x).all() for x in
                    (self.f_mu, self.p_mu, self.model.clock_mu, *scales))
                or not all((x > 0).all() for x in scales)):
            raise ValueError('checkpoint normalization must be finite with positive scales')

    @torch.no_grad()
    def predict(self, initial_fields, parameters, time):
        prof = torch.as_tensor(initial_fields, device=self.device, dtype=torch.float32)
        par = torch.as_tensor(parameters, device=self.device, dtype=torch.float32)
        t = torch.as_tensor(time, device=self.device, dtype=torch.float32)
        if prof.ndim != 3 or par.ndim != 2 or prof.shape[0] != par.shape[0]:
            raise ValueError('expected fields (batch, channels, space), parameters (batch, parameters)')
        if prof.shape[1] != self.f_mu.shape[1] or par.shape[1] != self.p_mu.shape[1]:
            raise ValueError('field or parameter count differs from training')
        if prof.shape[-1] != self.model.n_x:
            raise ValueError(f'clock encoder requires {self.model.n_x} spatial points')
        t = torch.broadcast_to(t, (prof.shape[0],))
        if not (torch.isfinite(prof).all() and torch.isfinite(par).all()
                and torch.isfinite(t).all()) or (t < 0).any():
            raise ValueError('inputs must be finite and time must be nonnegative')
        p, c = (prof-self.f_mu)/self.f_sd, (par-self.p_mu)/self.p_sd
        log_end = self.model.clock(p,c)*self.model.clock_sd+self.model.clock_mu
        terminal_time = torch.pow(10., log_end)
        if not torch.isfinite(terminal_time).all() or (terminal_time <= 0).any():
            raise FloatingPointError('predicted terminal time is not finite and positive')
        pred = self.model(p,c,t)*self.f_sd+self.f_mu
        # Avoid normalization roundoff weakening the exact initial-value contract.
        pred = torch.where((t == 0)[:,None,None], prof, pred)
        ratio = t/terminal_time
        if not torch.isfinite(pred).all() or not torch.isfinite(ratio).all():
            raise FloatingPointError('forecast fields or time ratio are not finite')
        return {'fields': pred.cpu().numpy(),
                'predicted_terminal_time': terminal_time.cpu().numpy(),
                'predicted_time_ratio': ratio.cpu().numpy(),
                'within_predicted_horizon': (ratio<=1).cpu().numpy()}
