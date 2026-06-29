from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class ActorCriticTerrainMlp(nn.Module):
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
        terrain_obs_dim=96,
        terrain_hidden_dims=[128],
        terrain_embedding_dim=64,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTerrainMlp.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.terrain_obs_dim = terrain_obs_dim
        self.terrain_embedding_dim = terrain_embedding_dim

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.actor_proprio_dim = num_actor_obs - terrain_obs_dim
        self.critic_proprio_dim = num_critic_obs - terrain_obs_dim
        if self.actor_proprio_dim <= 0 or self.critic_proprio_dim <= 0:
            raise ValueError(
                f"terrain_obs_dim incorrect, actor_proprio_dim={self.actor_proprio_dim}, "
                f"critic_proprio_dim={self.critic_proprio_dim}, actor={num_actor_obs}, critic={num_critic_obs}, "
                f"terrain_obs_dim={terrain_obs_dim}"
            )

        self.terrain_encoder = MLP(terrain_obs_dim, terrain_embedding_dim, terrain_hidden_dims, activation)

        actor_input_dim = self.actor_proprio_dim + terrain_embedding_dim
        critic_input_dim = self.critic_proprio_dim + terrain_embedding_dim

        self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.actor_proprio_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Terrain MLP: {self.terrain_encoder}")
        print(f"Actor MLP input_dim={actor_input_dim}: {self.actor}")

        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.critic_proprio_dim)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP input_dim={critic_input_dim}: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def _split_obs(self, obs):
        proprio_obs = obs[:, :-self.terrain_obs_dim]
        terrain_obs = obs[:, -self.terrain_obs_dim:]
        return proprio_obs, terrain_obs

    def _encode_obs(self, obs, normalizer):
        proprio_obs, terrain_obs = self._split_obs(obs)
        proprio_obs = normalizer(proprio_obs)
        terrain_embedding = self.terrain_encoder(terrain_obs)
        encoded_obs = torch.cat([proprio_obs, terrain_embedding], dim=-1)

        if torch.isnan(encoded_obs).any() or torch.isinf(encoded_obs).any():
            print(f"Warning: encoded_obs contains NaN or Inf: {encoded_obs}")

        return encoded_obs

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

    def update_distribution(self, obs):
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"Warning: obs contains NaN or Inf: {obs}")

        encoded_obs = self._encode_obs(obs, self.actor_obs_normalizer)
        mean = self.actor(encoded_obs)

        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print(f"Warning: mean contains NaN or Inf: {mean}")

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_obs = self.get_actor_obs(obs)
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_obs = self.get_actor_obs(obs)
        encoded_obs = self._encode_obs(actor_obs, self.actor_obs_normalizer)
        return self.actor(encoded_obs)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.get_critic_obs(obs)
        encoded_obs = self._encode_obs(critic_obs, self.critic_obs_normalizer)
        value = self.critic(encoded_obs)
        if torch.isnan(value).any() or torch.isinf(value).any():
            print(f"Warning: critic value contains NaN or Inf, {value}")
        return value

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            group_data = obs[obs_group]
            if torch.isnan(group_data).any() or torch.isinf(group_data).any():
                print(f"Warning: obs_group '{obs_group}' contains NaN or Inf: {group_data}")
            obs_list.append(group_data)
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            group_data = obs[obs_group]
            if torch.isnan(group_data).any() or torch.isinf(group_data).any():
                print(f"Warning: obs_group '{obs_group}' contains NaN or Inf: {group_data}")
            obs_list.append(group_data)
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            actor_proprio_obs, _ = self._split_obs(actor_obs)
            self.actor_obs_normalizer.update(actor_proprio_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            critic_proprio_obs, _ = self._split_obs(critic_obs)
            self.critic_obs_normalizer.update(critic_proprio_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
