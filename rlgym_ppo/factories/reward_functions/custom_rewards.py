from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

import numpy as np

class TouchVelChange(RewardFunction):
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            vel_difference = abs(np.linalg.norm(self.last_vel - state.ball.linear_velocity))
            reward = vel_difference / 4600.0

        self.last_vel = state.ball.linear_velocity

        return reward
    
class BallVelocityReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return (np.linalg.norm(state.ball.linear_velocity) / CAR_MAX_SPEED)**2
    

class PositiveVelocityPlayerToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        self.vel_reward = VelocityPlayerToBallReward()

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = self.vel_reward.get_reward(player, state, previous_action)
        pos_vel = max(0, vel)
        return (pos_vel)**2 
    
class PositiveVelocityBallToGoalReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        self.vel_reward = VelocityBallToGoalReward()

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = self.vel_reward.get_reward(player, state, previous_action)
        pos_vel = max(0, vel)
        return (pos_vel)**2