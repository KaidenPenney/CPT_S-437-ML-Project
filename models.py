import torch
import torch.nn as nn
from torch.distributions import Categorical
from ncps.torch import CfC

INPUT_SIZE = 12
HIDDEN_SIZE = 32
OUTPUT_SIZE = 3

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
        )
        self.actor = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.5]))

    def forward(self, observation):
        out = self.mlp(observation)
        
        return self.actor(out), self.critic(out).squeeze(-1)

    def get_action(self, observation_history):
        """Use this inside your training loop."""
        logits, critic_val = self.forward(observation_history)
        
        dist = Categorical(logits=logits)
        action_idx = dist.sample() 
        final_action = action_idx - 1
        
        return final_action, dist.log_prob(action_idx), dist.entropy(), critic_val

    def get_deterministic_action(self, observation_history):
        """Use this when testing, evaluating."""
        with torch.no_grad():
            logits, critic_val = self.forward(observation_history)
            final_action = torch.argmax(logits, dim=-1) - 1
            return final_action, critic_val


class RNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.actor = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.5]))

    def forward(self, observation_history):
        sequence_outputs, _ = self.rnn(observation_history)
        out = sequence_outputs[:, -1, :]

        return self.actor(out), self.critic(out).squeeze(-1)
    
    def get_action(self, observation_history):
        """Use this inside your training loop."""
        logits, critic_val = self.forward(observation_history)
        
        dist = Categorical(logits=logits)
        action_idx = dist.sample() 
        final_action = action_idx - 1
        
        return final_action, dist.log_prob(action_idx), dist.entropy(), critic_val

    def get_deterministic_action(self, observation_history):
        """Use this when testing, evaluating."""
        with torch.no_grad():
            logits, critic_val = self.forward(observation_history)
            final_action = torch.argmax(logits, dim=-1) - 1
            return final_action, critic_val


class LNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.cfc = CfC(
            input_size=INPUT_SIZE,
            units=HIDDEN_SIZE,
            return_sequences=False,
        )
        self.actor = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.5]))

    def forward(self, observation_history):
        batch_size = observation_history.shape[0]
        initial_state = observation_history.new_zeros(batch_size, HIDDEN_SIZE)
        out, _ = self.cfc(observation_history, initial_state)
        
        return self.actor(out), self.critic(out).squeeze(-1)
    
    def get_action(self, observation_history):
        """Use this inside your training loop."""
        logits, critic_val = self.forward(observation_history)
        
        dist = Categorical(logits=logits)
        action_idx = dist.sample() 
        final_action = action_idx - 1
        
        return final_action, dist.log_prob(action_idx), dist.entropy(), critic_val

    def get_deterministic_action(self, observation_history):
        """Use this when testing, evaluating."""
        with torch.no_grad():
            logits, critic_val = self.forward(observation_history)
            final_action = torch.argmax(logits, dim=-1) - 1
            return final_action, critic_val
        