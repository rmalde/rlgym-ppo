"""
File: discrete_policy.py
Author: Ronak Malde

Description:
    An implementation of a feed-forward neural network which parametrizes a categorical distribution over a space of actions.
"""


from torch.distributions import Categorical
import torch.nn as nn
import torch
import numpy as np


class LookupPolicy(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device

        assert len(layer_sizes) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], n_actions))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers).to(self.device)

        self.n_actions = n_actions

    def get_output(self, obs):
        t = type(obs)
        if t != torch.Tensor:
            if t != np.array:
                obs = np.asarray(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        return self.model(obs)

    def get_action(self, obs, deterministic=False):
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs: Observation to act on.
        :param deterministic: Whether the action should be chosen deterministically.
        :return: Chosen action and its logprob.
        """
        print("obs shape: ", obs.shape)
        probs = self.get_output(obs)
        print("probs shape: ", probs.shape)

        if deterministic:
            return probs.argmax().cpu().item(), 1

        distribution = Categorical(probs)
        action_idx = distribution.sample()
        log_prob = distribution.log_prob(action_idx)

        print("action idx: ", action_idx)
        return action_idx.cpu(), log_prob.cpu()

    def get_backprop_data(self, obs, acts):
        """
        Function to compute the data necessary for backpropagation.
        :param obs: Observations to pass through the policy.
        :param acts: Actions taken by the policy.
        :return: Action log probs & entropy.
        """
        acts = acts.long()
        print("backprop acts shape: ", acts.shape)
        probs = self.get_output(obs)
        distribution = Categorical(probs)

        entropy = distribution.entropy().to(self.device)
        log_probs = distribution.log_prob(acts).to(self.device)
        
        return log_probs, entropy.mean()