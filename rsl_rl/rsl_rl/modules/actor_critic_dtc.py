"""Pure-MLP actor-critic for DTC-lite.

Differs from ``ActorCriticTerrainMlp`` in that there is **no** terrain
encoder sub-module: with cost-based foothold selection in the planner,
the policy receives compact planner outputs + small local patches and
does not need a separate encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class ActorCriticDTC(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticDTC.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.obs_groups = obs_groups

        num_actor_obs = 0
        for g in obs_groups["policy"]:
            assert len(obs[g].shape) == 2, "ActorCriticDTC only supports 1D obs."
            num_actor_obs += obs[g].shape[-1]
        num_critic_obs = 0
        for g in obs_groups["critic"]:
            assert len(obs[g].shape) == 2, "ActorCriticDTC only supports 1D obs."
            num_critic_obs += obs[g].shape[-1]

        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        print(f"ActorCriticDTC actor input_dim={num_actor_obs}: {self.actor}")
        print(f"ActorCriticDTC critic input_dim={num_critic_obs}: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _cat(self, obs, group_key):
        parts = []
        for g in self.obs_groups[group_key]:
            x = obs[g]
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: obs group '{g}' has NaN/Inf")
            parts.append(x)
        return torch.cat(parts, dim=-1)

    def get_actor_obs(self, obs):
        return self._cat(obs, "policy")

    def get_critic_obs(self, obs):
        return self._cat(obs, "critic")

    def update_distribution(self, obs):
        x = self.actor_obs_normalizer(obs)
        mean = self.actor(x)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        a = self.get_actor_obs(obs)
        self.update_distribution(a)
        return self.distribution.sample()

    def act_inference(self, obs):
        a = self.get_actor_obs(obs)
        return self.actor(self.actor_obs_normalizer(a))

    def evaluate(self, obs, **kwargs):
        c = self.get_critic_obs(obs)
        return self.critic(self.critic_obs_normalizer(c))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
