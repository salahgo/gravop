"""Scientific intervention guards for the separate F10 development model."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from forecast_factorial import ARMS, FactorialOperator, ClockConditionedOperator


@pytest.mark.verification
@pytest.mark.parametrize('fields', [1, 2])
def test_full_reproduces_original_and_every_arm_preserves_initial_field(fields):
    torch.set_num_threads(1)
    torch.manual_seed(317)
    cfg = dict(n_fields=fields, n_par=2, n_x=128, width=8, modes=4, n_layers=1)
    original = ClockConditionedOperator(**cfg).eval()
    p, c = torch.randn(4, fields, 128), torch.randn(4, 2)
    t = torch.tensor([0., .01, .9, 2.])
    for arm in ARMS:
        model = FactorialOperator(arm, **cfg).eval()
        model.load_state_dict(original.state_dict())
        assert sum(x.numel() for x in model.parameters()) == sum(x.numel() for x in original.parameters())
        torch.testing.assert_close(model(p, c, torch.zeros(4)), p, rtol=0, atol=0)
        if arm == 'full':
            torch.testing.assert_close(model(p, c, t), original(p, c, t), rtol=0, atol=0)
        model(p, c, t).square().mean().backward()
        assert all(x.grad is None for x in model.clock.parameters())
        assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)


@pytest.mark.verification
def test_direct_arm_is_invariant_to_clock_intervention():
    torch.manual_seed(318)
    model = FactorialOperator('direct', n_fields=1, n_par=2, width=8, modes=4, n_layers=1).eval()
    p, c, t = torch.randn(3, 1, 128), torch.randn(3, 2), torch.tensor([.1, .5, 1.])
    before = model(p, c, t).detach()
    model.clock_mu.add_(1.)
    torch.testing.assert_close(model(p, c, t), before, rtol=0, atol=0)
    for arm in ('full', 'features_only', 'gate_only'):
        model.arm = arm
        before = model(p, c, t).detach()
        model.clock_mu.add_(1.)
        assert not torch.equal(model(p, c, t), before)
