"""
Actor-critic networks for the maze agent.

Every model maps a sequence of observations to action logits and a value estimate,
and carries a hidden state between agent decisions. The recurrent models really
use that state across time steps (the MLP ignores it), and the state is reset at
the start of each episode.

Each architecture gets a hidden size picked so its total parameter count
(including the actor and critic heads) is as close as possible to a shared budget.
"""
import torch
import torch.nn as nn
from ncps.torch import CfCCell

ARCHITECTURES = ("mlp", "rnn", "cfc", "gru", "lstm")
DEFAULT_PARAM_BUDGET = 6000


class ActorCritic(nn.Module):
    def __init__(self, obs_size, num_actions, hidden_size):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self._init_core()
        self.action_net = nn.Linear(hidden_size, num_actions)
        self.value_net = nn.Linear(hidden_size, 1)
        # Small initial logits so the starting policy is close to uniform.
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.zeros_(self.action_net.bias)

    state_size = 0

    def _init_core(self):
        raise NotImplementedError

    def _core_step(self, obs, state):
        """One time step: (B, obs_size), (B, state_size) -> features (B, hidden), new state."""
        raise NotImplementedError

    def initial_state(self, batch_size, device="cpu"):
        return torch.zeros(batch_size, self.state_size, device=device)

    def forward(self, obs, state, episode_starts):
        """
        obs: (T, B, obs_size), state: (B, state_size), episode_starts: (T, B), 1.0 where
        obs[t] is the first observation of a new episode (state is zeroed before it).
        Returns logits (T, B, A), values (T, B), final state (B, state_size).
        """
        features = []
        for t in range(obs.shape[0]):
            state = state * (1.0 - episode_starts[t]).unsqueeze(-1)
            out, state = self._core_step(obs[t], state)
            features.append(out)
        features = torch.stack(features)
        return self.action_net(features), self.value_net(features).squeeze(-1), state


class MLP(ActorCritic):
    """Feedforward baseline: no memory, decides from the current observation only."""

    def _init_core(self):
        self.core = nn.Sequential(
            nn.Linear(self.obs_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        )

    def forward(self, obs, state, episode_starts):
        features = self.core(obs)
        return self.action_net(features), self.value_net(features).squeeze(-1), state


class RNN(ActorCritic):
    """Vanilla (Elman) tanh RNN: memory without gating or liquid time constants."""

    def _init_core(self):
        self.state_size = self.hidden_size
        self.cell = nn.RNNCell(self.obs_size, self.hidden_size)

    def _core_step(self, obs, state):
        state = self.cell(obs, state)
        return state, state


class GRU(ActorCritic):
    """Optional gated baseline (not in the original proposal)."""

    def _init_core(self):
        self.state_size = self.hidden_size
        self.cell = nn.GRUCell(self.obs_size, self.hidden_size)

    def _core_step(self, obs, state):
        state = self.cell(obs, state)
        return state, state


class LSTM(ActorCritic):
    """Optional gated baseline (not in the original proposal). State is [h, c]."""

    def _init_core(self):
        self.state_size = 2 * self.hidden_size
        self.cell = nn.LSTMCell(self.obs_size, self.hidden_size)

    def _core_step(self, obs, state):
        h, c = self.cell(obs, (state[:, :self.hidden_size], state[:, self.hidden_size:]))
        return h, torch.cat([h, c], dim=-1)


class CfC(ActorCritic):
    """
    Closed-form Continuous-time liquid network (Hasani et al., 2022) from the ncps
    package, with one backbone layer the same width as the state. Every agent
    decision is the same length of game time, so the elapsed time ts is 1.0.
    """

    def _init_core(self):
        self.state_size = self.hidden_size
        self.cell = CfCCell(
            self.obs_size,
            self.hidden_size,
            backbone_units=self.hidden_size,
            backbone_layers=1,
        )

    def _core_step(self, obs, state):
        out, state = self.cell(obs, state, 1.0)
        return out, state


MODEL_CLASSES = {"mlp": MLP, "rnn": RNN, "cfc": CfC, "gru": GRU, "lstm": LSTM}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def hidden_size_for_budget(arch, obs_size, num_actions, param_budget):
    """Hidden size whose total parameter count is closest to param_budget."""
    model_class = MODEL_CLASSES[arch]
    best_size, best_gap = None, None
    for hidden_size in range(2, 1024):
        params = count_parameters(model_class(obs_size, num_actions, hidden_size))
        gap = abs(params - param_budget)
        if best_gap is None or gap < best_gap:
            best_size, best_gap = hidden_size, gap
        if params > param_budget:
            break
    return best_size


def build_model(arch, obs_size, num_actions, param_budget=DEFAULT_PARAM_BUDGET, hidden_size=None):
    if hidden_size is None:
        hidden_size = hidden_size_for_budget(arch, obs_size, num_actions, param_budget)
    return MODEL_CLASSES[arch](obs_size, num_actions, hidden_size)
