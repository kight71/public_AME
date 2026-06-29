"""Adapter from Isaac Lab 2.3.0's RslRlVecEnvWrapper to the TensorDict-based
VecEnv API expected by the bundled rsl_rl runner (newer fork).

Isaac Lab 2.3.0:
    get_observations()  -> (policy_obs: Tensor, extras: dict)
    step(actions)       -> (policy_obs: Tensor, rew, dones, extras: dict)
    extras["observations"] holds the full multi-group obs dict.

Bundled rsl_rl runner expects:
    get_observations()  -> TensorDict({"policy": ..., "critic": ..., ...})
    step(actions)       -> (TensorDict, rew, dones, extras)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


class IsaacLabTensorDictAdapter:
    """Wraps Isaac Lab 2.3.0's RslRlVecEnvWrapper to expose the TensorDict API."""

    def __init__(self, inner):
        self._inner = inner

    # ---------- attribute / property delegation ----------

    def __getattr__(self, name):
        # only called for attributes not found on self
        return getattr(self._inner, name)

    @property
    def num_envs(self) -> int:
        return self._inner.num_envs

    @property
    def device(self):
        return self._inner.device

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._inner.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self._inner.episode_length_buf = value

    @property
    def cfg(self):
        return self._inner.cfg

    @property
    def unwrapped(self):
        return self._inner.unwrapped

    @property
    def max_episode_length(self):
        return self._inner.max_episode_length

    # ---------- observation API conversion ----------

    @staticmethod
    def _to_tensordict(extras: dict, batch_size: int) -> TensorDict:
        obs_dict = extras.get("observations", {}) if isinstance(extras, dict) else {}
        # Filter non-tensor or empty entries.
        clean = {k: v for k, v in obs_dict.items() if isinstance(v, torch.Tensor)}
        if not clean:
            raise RuntimeError(
                "extras['observations'] contains no tensor groups; nothing to wrap."
            )
        return TensorDict(clean, batch_size=[batch_size])

    def get_observations(self) -> TensorDict:
        _policy, extras = self._inner.get_observations()
        return self._to_tensordict(extras, self._inner.num_envs)

    def reset(self):
        _policy, extras = self._inner.reset()
        return self._to_tensordict(extras, self._inner.num_envs), extras

    def step(self, actions: torch.Tensor):
        _policy, rew, dones, extras = self._inner.step(actions)
        return (
            self._to_tensordict(extras, self._inner.num_envs),
            rew,
            dones,
            extras,
        )
