from __future__ import annotations

from src.evaluation import collector as collector_mod
from src.evaluation.types import CompositeEvalResult


class _DummyPolicy:
    def state_dict(self) -> dict[str, object]:
        return {}


def test_simulation_eval_collector_tracks_tianshou_stats(monkeypatch) -> None:
    def _fake_eval(**_: object) -> CompositeEvalResult:
        return CompositeEvalResult(
            policy_result=None,
            replay_result=None,
            summary={"score_mean": 0.5},
        )

    monkeypatch.setattr(collector_mod, "evaluate_composite_with_simulator", _fake_eval)

    collector = collector_mod.SimulationEvalCollector(
        policy=_DummyPolicy(),
        sim_eval_cfg={"enabled": True},
        action_max=1.0,
        eval_every_n_epoch=1,
    )
    collector.set_epoch(1)

    stats = collector.collect(n_episode=2)
    assert stats.n_collected_steps == 2
    assert collector.collect_step == 2
    assert collector.collect_episode == 2
    assert collector.collect_time > 0.0

    collector.reset(reset_stats=True)
    assert collector.collect_step == 0
    assert collector.collect_episode == 0
    assert collector.collect_time == 0.0
