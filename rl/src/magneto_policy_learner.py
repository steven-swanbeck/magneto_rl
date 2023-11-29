#!/usr/bin/env python3
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from typing import Tuple, Callable
import torch.nn as nn

class ModelLearnerNetwork (torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Policy network
        # Your policy_net must take in a vector of length feature_dim
        # and ouput a vector of length last_layer_dim_pi
        
        # self.policy_net = torch.nn.Linear(feature_dim, last_layer_dim_pi)
        
        # # & decent
        # self.policy_net = torch.nn.Sequential(
        #     torch.nn.Linear(feature_dim, last_layer_dim_pi),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
        # )
        
        # self.policy_net = torch.nn.Sequential(
        #     torch.nn.Linear(feature_dim, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, last_layer_dim_pi),
        # )
        
        self.policy_net = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=feature_dim,
                hidden_size=last_layer_dim_pi,
            ),
            'linear': nn.Linear(
                in_features=last_layer_dim_pi,
                out_features=last_layer_dim_pi,
            ),
            # 'activation': nn.Tanh(),
        })

        # Value network
        # Your value_net must take in a vector of length feature_dim
        # and ouput a vector of length last_layer_dim_vf
        
        # self.value_net = torch.nn.Linear(feature_dim, last_layer_dim_vf)
        
        # # & decent
        # self.value_net = torch.nn.Sequential(
        #     torch.nn.Linear(feature_dim, last_layer_dim_vf),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(last_layer_dim_vf, last_layer_dim_vf),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(last_layer_dim_vf, last_layer_dim_vf),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(last_layer_dim_vf, last_layer_dim_vf),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(last_layer_dim_vf, last_layer_dim_vf),
        # )
        
        # self.value_net = torch.nn.Sequential(
        #     torch.nn.Linear(feature_dim, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, last_layer_dim_vf),
        # )
        
        self.value_net = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=feature_dim,
                hidden_size=last_layer_dim_vf,
            ),
            'linear': nn.Linear(
                in_features=last_layer_dim_vf,
                out_features=last_layer_dim_vf,
            ),
            # 'activation': nn.Tanh(),
        })
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        out_pi, _ = self.policy_net['lstm'](features)
        out_pi = self.policy_net['linear'](out_pi)
        return out_pi
        # return self.policy_net['activation'](out_pi)
        # return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        out_vf, _ = self.value_net['lstm'](features)
        out_vf = self.value_net['linear'](out_vf)
        return out_vf
        # return self.value_net['activation'](out_vf)
        # return self.value_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ModelLearnerNetwork(self.features_dim)
