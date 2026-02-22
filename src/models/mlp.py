"""Default MLP model factories for continuous control."""

from __future__ import annotations

from typing import Any

from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorDeterministic, ContinuousCritic

from src.models.base import ModelFactory


class MLPActorCriticFactory(ModelFactory):
    """Builds deterministic actor and twin critics with shared MLP recipe."""

    def build_actor(self, cfg: Any, space_info: Any, device: str):
        net_a = Net(
            state_shape=space_info.observation_info.obs_shape,
            hidden_sizes=list(cfg.hidden_sizes),
        )
        actor = ContinuousActorDeterministic(
            preprocess_net=net_a,
            action_shape=space_info.action_info.action_shape,
            max_action=space_info.action_info.max_action,
        ).to(device)
        return actor

    def build_critic(self, cfg: Any, space_info: Any, device: str):
        net_c1 = Net(
            state_shape=space_info.observation_info.obs_shape,
            action_shape=space_info.action_info.action_shape,
            hidden_sizes=list(cfg.hidden_sizes),
            concat=True,
        )
        net_c2 = Net(
            state_shape=space_info.observation_info.obs_shape,
            action_shape=space_info.action_info.action_shape,
            hidden_sizes=list(cfg.hidden_sizes),
            concat=True,
        )
        critic1 = ContinuousCritic(preprocess_net=net_c1).to(device)
        critic2 = ContinuousCritic(preprocess_net=net_c2).to(device)
        return critic1, critic2


class MLPActorOnlyFactory(ModelFactory):
    """Builds only actor network for behavior cloning style algorithms."""

    def build_actor(self, cfg: Any, space_info: Any, device: str):
        net = Net(
            state_shape=space_info.observation_info.obs_shape,
            hidden_sizes=list(cfg.hidden_sizes),
        )
        actor = ContinuousActorDeterministic(
            preprocess_net=net,
            action_shape=space_info.action_info.action_shape,
            max_action=space_info.action_info.max_action,
        ).to(device)
        return actor

    def build_critic(self, cfg: Any, space_info: Any, device: str):
        return None, None
