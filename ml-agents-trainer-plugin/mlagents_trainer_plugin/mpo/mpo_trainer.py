# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (MPO)
# Contains an implementation of MPO as described in: https://arxiv.org/abs/1707.06347

from typing import cast, Type, Union, Dict, Any

import numpy as np

from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
# from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer
from mlagents.trainers.trainer.off_policy_trainer import OffPolicyTrainer
from mlagents.trainers.policy.policy import Policy
from mlagents.trainers.trainer.trainer_utils import get_gae
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from .mpo_optimizer_torch import TorchMPOOptimizer, MPOSettings
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings

from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic

logger = get_logger(__name__)

TRAINER_NAME = "mpo"


class MPOTrainer(OffPolicyTrainer):
    """The MPOTrainer is an implementation of the MPO algorithm."""

    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        """
        Responsible for collecting experiences and training MPO model.
        :param behavior_name: The name of the behavior associated with trainer config
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_settings: The parameters for the trainer.
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param artifact_path: The directory within which to store artifacts from this trainer.
        """
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )
        self.hyperparameters: MPOSettings = cast(
            MPOSettings, self.trainer_settings.hyperparameters
        )
        self.seed = seed
        self.policy: TorchPolicy = None  # type: ignore

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the update buffer.
        Processing involves calculating value and advantage targets for model updating step.
        :param trajectory: The Trajectory tuple containing the steps to be processed.
        """
        super()._process_trajectory(trajectory)
        agent_id = trajectory.agent_id  # All the agents should have the same ID

        agent_buffer_trajectory = trajectory.to_agentbuffer()
        # Check if we used group rewards, warn if so.
        self._warn_if_group_reward(agent_buffer_trajectory)

        # Update the normalization
        if self.is_training:
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            self.optimizer.critic.update_normalization(agent_buffer_trajectory)

        # Get all value estimates
        (
            value_estimates,
            value_next,
            value_memories,
        ) = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory,
            trajectory.next_obs,
            trajectory.done_reached and not trajectory.interrupted,
        )
        if value_memories is not None:
            agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)

        for name, v in value_estimates.items():
            agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(
                v
            )
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
                np.mean(v),
            )

        # Evaluate all reward functions
        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )
            agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(
                evaluate_result
            )
            # Report the reward signals
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        self._append_to_update_buffer(agent_buffer_trajectory)

        # If this was a terminal trajectory, append stats and reset reward collection
        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

    def create_optimizer(self) -> TorchOptimizer:
        return TorchMPOOptimizer(  # type: ignore
            cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
        )  # type: ignore

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and MPO hyperparameters
        :param parsed_behavior_id:
        :param behavior_spec: specifications for policy construction
        :return policy
        """
        actor_cls: Union[Type[SimpleActor], Type[SharedActorCritic]] = SimpleActor
        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,
        }

        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
        )
        return policy

    def get_policy(self, name_behavior_id: str) -> Policy:
        """
        Gets policy from trainer associated with name_behavior_id
        :param name_behavior_id: full identifier of policy
        """

        return self.policy


    @staticmethod
    def get_settings_type():
        return MPOSettings

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME

def get_type_and_setting():
    return {MPOTrainer.get_trainer_name(): MPOTrainer}, {
        MPOTrainer.get_trainer_name(): MPOSettings
    }