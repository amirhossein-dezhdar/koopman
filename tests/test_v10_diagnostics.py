import math
import numpy as np

from cr_koop_mpc.models.conformal import ConformalResidual
from cr_koop_mpc.triggering.event_trigger import ResidualDrivenTrigger, TriggerReason
from cr_koop_mpc.utils import load_config


def test_trigger_age_is_pre_reset_and_pure_stale():
    trig = ResidualDrivenTrigger(
        sigma_min=0.1, sigma_max=1.0, lambda_w=0.0,
        eps_safety=-10.0, eps_filter=10.0, N_max=3,
    )
    d1 = trig.step(np.zeros(1), np.zeros(1), 0.0, 0.0, 0.0)
    d2 = trig.step(np.zeros(1), np.zeros(1), 0.0, 0.0, 0.0)
    d3 = trig.step(np.zeros(1), np.zeros(1), 0.0, 0.0, 0.0)
    assert not d1.fire and d1.age == 1
    assert not d2.fire and d2.age == 2
    assert d3.fire and d3.age == 3
    assert d3.reason == TriggerReason.STALE
    assert d3.pure_stale and not d3.nonstale


def test_conformal_logs_actual_previous_bound_coverage():
    conf = ConformalResidual(alpha=0.1, adaptive=True, gamma=0.02, window=20)
    w0 = conf.calibrate(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    conf.update(0.05)
    assert conf.last_bound_before_update == w0
    assert conf.last_covered == 1.0
    conf.update(10.0)
    assert conf.last_miscovered == 1.0


def test_config_inheritance_event_pure():
    cfg = load_config("configs/event_pure.yaml")
    assert cfg.trigger.N_max == 999999
    assert cfg.env.dt == 0.05
    assert cfg.residual.state_indices == [1, 4]
