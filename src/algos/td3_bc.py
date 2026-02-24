"""TD3-BC algorithm factory."""

from __future__ import annotations

from typing import Any

from tianshou.algorithm import TD3BC
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.exploration import GaussianNoise

from src.algos.base import AlgoFactory
from src.core.exceptions import ConfigurationError
from src.core.types import ModelBundle


class TD3BCFactory(AlgoFactory):
    """Builds TD3-BC with deterministic actor and twin critics."""

    def build(self, cfg: Any, env: Any, model_bundle: ModelBundle, device: str):
        if model_bundle.critic1 is None or model_bundle.critic2 is None:
            raise ConfigurationError("td3_bc requires model with twin critics.")

        policy = ContinuousDeterministicPolicy(
            actor=model_bundle.actor,
            exploration_noise=GaussianNoise(sigma=float(cfg.exploration_noise)),
            action_space=env.action_space,
            action_scaling=False,
            action_bound_method=None,
        )

        algo = TD3BC(
            policy=policy,
            policy_optim=AdamOptimizerFactory(lr=float(cfg.actor_lr)),
            critic=model_bundle.critic1,
            critic_optim=AdamOptimizerFactory(lr=float(cfg.critic_lr)),
            critic2=model_bundle.critic2,
            critic2_optim=AdamOptimizerFactory(lr=float(cfg.critic_lr)),
            tau=float(cfg.tau),
            gamma=float(cfg.gamma),
            policy_noise=float(cfg.policy_noise),
            update_actor_freq=int(cfg.update_actor_freq),
            noise_clip=float(cfg.noise_clip),
            alpha=float(cfg.alpha),
            n_step_return_horizon=int(cfg.n_step),
        )
        return algo
