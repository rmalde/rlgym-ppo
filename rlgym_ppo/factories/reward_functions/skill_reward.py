import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np

from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.reward_functions.common_rewards import EventReward
from rlgym_sim.utils.gamestates import GameState, PlayerData

from rlgym_ppo.models import FFN, SkillMask
from rlgym_ppo.factories.obs_builders.pyr_obs import PyrObs

MODEL_PATH = "rlgym_ppo/factories/reward_functions/low_skill_reward.pt"

class SkillReward(RewardFunction):
    def __init__(self, obs_size, action_size, layer_sizes, model_config, max_steps, device):
        super().__init__()
        skill_mask = SkillMask()
        self.skill_model = nn.Sequential(skill_mask, FFN(
            skill_mask.out_dim,
            action_size,
            layer_sizes,
            objective="regression",
            config=model_config,
        )).to(device).eval()
        self.skill_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        self.obs_builder = PyrObs()

        self.device = device
        self.max_steps = max_steps

        self.step_penalty = 1.

        self.win_event = EventReward(team_goal=1)
        self.lose_event = EventReward(concede=1)

    def reset(self, initial_state: GameState):
        self.running_sums = defaultdict(float)
        self.win_event.reset(initial_state)
        self.lose_event.reset(initial_state)
        self.time = 0

    def pre_step(self, state: GameState):
        self.time += 1

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        obs = self.obs_builder.build_obs(player, state, previous_action)
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        skill_reward = 0
        with torch.no_grad():
            skill_reward = float(self.skill_model(obs).item())
        self.running_sums[player.car_id] += skill_reward

        win = self.win_event.get_reward(player, state, previous_action)
        loss = self.lose_event.get_reward(player, state, previous_action)

        reward = skill_reward - self.step_penalty
        if win == 1:
            reward += self.max_steps # score bonus

        if loss == 1:
            avg_skill = self.running_sums[player.car_id] / self.time
            remaining_time = self.max_steps - self.time
            imputed_reward = (avg_skill - self.step_penalty) * remaining_time # should be negative
            reward += imputed_reward
            reward -= self.max_steps # concede penalty

        return reward

