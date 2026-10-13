import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from ncps.torch import CfC
from stable_baselines3.common.policies import BasePolicy

INPUT_SIZE = 15
HIDDEN_SIZE = 32
OUTPUT_SIZE = 3

# This function is really patchy so that it functions with stable_baselines3 PPO
# I did this because I think it is cleaner than making our own PPO algorithm function
class BaseMazePolicy(BasePolicy):
    """Handles Policy Info So That Can Be Used With stable_baselines3 PPO"""
    def __init__(self, observation_space, action_space, lr_schedule=None, **kwargs):

        kwargs.pop("use_sde", None)
        kwargs.pop("net_arch", None)
        kwargs.pop("activation_fn", None)
        super().__init__(observation_space, action_space, **kwargs)

        self._init_network_layers()
        
        self.action_net = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.value_net = nn.Linear(HIDDEN_SIZE, 1)
        
        if lr_schedule is not None:
            init_lr = lr_schedule(1.0)
            self.optimizer = optim.Adam(self.parameters(), lr=init_lr)

    def _init_network_layers(self) -> None:
        raise NotImplementedError

    def _extract_latent_features(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        latent = self._extract_latent_features(obs)
        action_logits = self.action_net(latent)
        critic_value = self.value_net(latent)

        distribution = Categorical(logits=action_logits)
        actions = torch.argmax(action_logits, dim=-1) if deterministic else distribution.sample()
        return actions, critic_value, distribution.log_prob(actions)

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        latent = self._extract_latent_features(observation)
        action_logits = self.action_net(latent)
        if deterministic:
            return torch.argmax(action_logits, dim=-1)
        return Categorical(logits=action_logits).sample()

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self._extract_latent_features(obs)
        return self.value_net(latent)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        latent = self._extract_latent_features(obs)
        distribution = Categorical(logits=self.action_net(latent))
        return self.value_net(latent), distribution.log_prob(actions), distribution.entropy()


class MLP(BaseMazePolicy):
    def _init_network_layers(self) -> None:
        self.mlp_extractor = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
        )

    def _extract_latent_features(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 3:
            obs = obs[:, -1, :]
        return self.mlp_extractor(obs)


class RNN(BaseMazePolicy):
    """Recurrent Strategy is Baked in So Works Standard with PPO"""
    def _init_network_layers(self) -> None:
        self.lstm_actor = nn.RNN(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)

    def _extract_latent_features(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)
        sequence_outputs, _ = self.lstm_actor(obs)
        return sequence_outputs[:, -1, :]


class LNN(BaseMazePolicy):
    """Recurrent Strategy is Baked in So Works Standard with PPO"""
    def _init_network_layers(self) -> None:
        self.lstm_actor = CfC(input_size=INPUT_SIZE, units=HIDDEN_SIZE, return_sequences=False)

    def _extract_latent_features(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)
        initial_state = obs.new_zeros(obs.shape[0], HIDDEN_SIZE)
        out, _ = self.lstm_actor(obs, initial_state)
        return out
