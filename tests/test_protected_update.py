"""Exercise the registered displacement projection and nonlinear loss check on native MLX."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from loopqwen.research import followup_fit
from loopqwen.worker import acquire


class Plane(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = mx.array([1.0, 0.0])


@pytest.mark.parametrize("speed,expected", [(0.5, 1.0), (4.0, 0.25), (64.0, 0.0)])
def test_nonlinear_protection(monkeypatch, speed, expected):
    # Along the unprotected coordinate, curvature can require a smaller or rolled-back step.
    acquire()
    net = Plane()
    initial = net.point
    monkeypatch.setattr(
        followup_fit, "reference_loss", lambda model: mx.sum(model.point**2)
    )
    adam = optim.AdamW(speed, weight_decay=0)
    measured = followup_fit.protected_update(
        net,
        adam,
        {"point": mx.array([1.0, -1.0])},
        {"point": mx.array([2.0, 0.0])},
        (),
        1.0,
        0.00005,
    )
    assert measured["factor"] == expected and not measured["projected"]
    assert measured["after"] <= 1.00005
    assert measured["applied_step_l2"] == pytest.approx(
        mx.linalg.norm(net.point - initial).item()
    )
    assert adam.state["step"].item() == 1


def test_conflict_projects_optimizer_displacement(monkeypatch):
    acquire()
    net = Plane()
    monkeypatch.setattr(followup_fit, "reference_loss", lambda model: model.point[0])
    optimizer = optim.AdamW(0.25, weight_decay=0)
    measured = followup_fit.protected_update(
        net,
        optimizer,
        {"point": mx.array([-1.0, 1.0])},
        {"point": mx.array([1.0, 0.0])},
        (),
        1.0,
        0.0,
    )
    assert measured["projected"] and measured["raw_directional"] > 0
    assert measured["factor"] == 1 and measured["actual_directional"] == 0
    assert net.point[0].item() == 1 and net.point[1].item() < 0
