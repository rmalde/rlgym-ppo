"""
File: discrete_policy.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which parametrizes a discrete distribution over a space of actions.
"""

from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from rlgym_ppo.models import FFN


class DiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device

        assert (
            len(layer_sizes) != 0
        ), "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"

        self.model = FFN(
            input_shape, n_actions, layer_sizes, objective="classification"
        ).to(device)

        self.n_actions = n_actions

    def get_output(self, obs):
        t = type(obs)
        if t != torch.Tensor:
            if t != np.array:
                obs = np.asarray(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits = self.model(obs)
        return F.softmax(logits, dim=-1)

    def get_action(self, obs, deterministic=False):
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs: Observation to act on.
        :param deterministic: Whether the action should be chosen deterministically.
        :return: Chosen action and its logprob.
        """

        probs = self.get_output(obs)
        probs = probs.view(-1, self.n_actions)
        probs = torch.clamp(probs, min=1e-11, max=1)

        if deterministic:
            return probs.cpu().numpy().argmax(), 0

        action = torch.multinomial(probs, 1, True)
        log_prob = torch.log(probs).gather(-1, action)

        return action.flatten().cpu(), log_prob.flatten().cpu()

    def get_backprop_data(self, obs, acts):
        """
        Function to compute the data necessary for backpropagation.
        :param obs: Observations to pass through the policy.
        :param acts: Actions taken by the policy.
        :return: Action log probs & entropy.
        """
        acts = acts.long()
        probs = self.get_output(obs)
        probs = probs.view(-1, self.n_actions)
        probs = torch.clamp(probs, min=1e-11, max=1)

        log_probs = torch.log(probs)
        action_log_probs = log_probs.gather(-1, acts)
        entropy = -(log_probs * probs).sum(dim=-1)

        return action_log_probs.to(self.device), entropy.to(self.device).mean()
