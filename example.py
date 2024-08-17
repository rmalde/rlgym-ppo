import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger


OBS_SIZE = 111
ACTION_SIZE = 90

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [
            game_state.players[0].car_data.linear_velocity,
            game_state.players[0].car_data.rotation_mtx(),
            game_state.orange_score,
        ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {
            "x_vel": avg_linvel[0],
            "y_vel": avg_linvel[1],
            "z_vel": avg_linvel[2],
            "Cumulative Timesteps": cumulative_timesteps,
        }
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import (
        VelocityPlayerToBallReward,
        VelocityBallToGoalReward,
        EventReward,
        FaceBallReward,
    )

    # from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        NoTouchTimeoutCondition,
        GoalScoredCondition,
        TimeoutCondition,
    )
    from rlgym_sim.utils import common_values
    # from rlgym_sim.utils.action_parsers import ContinuousAction
    # from rlgym_sim.utils.state_setters import RandomState, DefaultState

    from rlgym_ppo.factories.obs_builders import PyrObs
    from rlgym_ppo.factories.action_parsers import LookupAction
    from rlgym_ppo.factories.state_setters import SemiRandomState
    from rlgym_ppo.factories.reward_functions import SkillReward

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 4
    timeout_seconds = 15
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))
    # timeout_ticks = 450

    # action_parser = ContinuousAction()
    action_parser = LookupAction()
    terminal_conditions = [
        TimeoutCondition(timeout_ticks),
        GoalScoredCondition(),
    ]

    rewards_to_combine = (
        EventReward(touch=1),
        VelocityPlayerToBallReward(),
        FaceBallReward(),
    )
    reward_weights = (1.0, 5.0, 1.0,)

    reward_fn = CombinedReward(
        reward_functions=rewards_to_combine, reward_weights=reward_weights
    )
    reward_model_config = {
        'layer_sizes': [1024, 1024, 1024, 1024],
        'use_batch_norm': True,
        "dropout": 0.2
    }
    reward_fn = SkillReward(
        obs_size=OBS_SIZE,
        action_size=ACTION_SIZE,
        layer_sizes=[1024, 1024, 1024, 1024],
        model_config=reward_model_config,
        max_steps=timeout_ticks,
        device='cpu',
    )

    obs_builder = PyrObs()
    state_setter = SemiRandomState(p_random=0.5)

    env = rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=team_size,
        spawn_opponents=spawn_opponents,
        terminal_conditions=terminal_conditions,
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        action_parser=action_parser,
        state_setter=state_setter,
    )

    return env


if __name__ == "__main__":
    from rlgym_ppo import Learner

    metrics_logger = ExampleLogger()

    # 32 processes
    n_proc = 64

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        build_rocketsim_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=metrics_logger,
        ppo_batch_size=50_000,
        ts_per_iteration=50_000,
        exp_buffer_size=100_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=0.01,
        ppo_epochs=2,
        policy_layer_sizes=[2048, 2048, 2048, 1024, 1024],
        critic_layer_sizes=[2048, 2048, 2048, 1024, 1024],
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=1_000_000,
        timestep_limit=int(1e69),
        log_to_wandb=True,
        wandb_project_name="skill-ppo",
        checkpoint_load_folder=None,
    )
    learner.learn()
