import numpy as np
import attr
from typing import List, Dict, Tuple, Optional, Union, Any, cast
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.networks import Actor, Critic, ValueNetwork, BranchValueNetwork
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from mlagents.trainers.settings import ScheduleType, NetworkSettings
from torch.distributions import Categorical
from mlagents_envs.logging_util import get_logger
from mlagents.torch_utils import torch, nn, default_device
from mlagents_envs.timers import timed
from mlagents_envs.base_env import ActionSpec, ObservationSpec
import copy
from scipy.optimize import minimize
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

logger = get_logger(__name__)
# TODO: fix saving to onnx

def bt(m):
    return m.transpose(dim0=-2, dim1=-1)

def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)

def categorical_kl(p1, p2):
    """
    calculates KL between two Categorical distributions
    :param p1: (B, D)
    :param p2: (B, D)
    """
    p1 = torch.clamp_min(p1, 0.0001)  # actually no need to clamp
    p2 = torch.clamp_min(p2, 0.0001)  # avoid zero division
    return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))


@attr.s(auto_attribs=True)
class MPOSettings(OffPolicyHyperparamSettings):
    dual_constraint: float = 0.1
    kl_constraint: float = 0.01 # constraint for discrete case (M-step)
    alpha_scale: float = 10.0 # scaling factor for lagrangian multiplier (M-step)
    batch_size: int = 256 # minibatch size
    mstep_iteration_num: int = 5 # Number of iterations for M-Step
    evaluate_episode_maxstep: int = 200 # Maximum evaluate steps of an episode
    alpha_max: float = 1.0
    actor_learning_rate: float = 0.001
    critic_learning_rate: float = 0.001
    buffer_size: int = 200000
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    gamma: float = 0.99

class TorchMPOOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The MPO optimizer has a value estimator and a loss function.
        :param policy: A TorchPolicy object that will be updated by this MPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.

        super().__init__(policy, trainer_settings)

        self.hyperparameters: MPOSettings = cast(
            MPOSettings, trainer_settings.hyperparameters
        )

        self.target_policy = copy.deepcopy(self.policy)

        self._critic = QNetwork(
            stream_names=self.reward_signals.keys(),
            observation_specs=policy.behavior_spec.observation_specs,
            network_settings=policy.network_settings,
            action_spec=policy.behavior_spec.action_spec,
        )
        self._critic.to(default_device())

        self._target_critic = QNetwork(
            stream_names=self.reward_signals.keys(),
            observation_specs=policy.behavior_spec.observation_specs,
            network_settings=policy.network_settings,
            action_spec=policy.behavior_spec.action_spec,
        )
        self._target_critic.to(default_device())

        self.norm_loss_q = nn.MSELoss()

        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=self.trainer_settings.hyperparameters.actor_learning_rate
        )

        self.critic_optimizer = torch.optim.Adam(
            self._critic.parameters(), lr=self.trainer_settings.hyperparameters.critic_learning_rate
        )

        self.ε_dual: float = self.hyperparameters.dual_constraint
        self.α: float = 0.0                # Langrangian multiplier for discrete action space (M-step)
        self.η: float = np.random.rand()   # Parameter to fit in closed form
        self.max_return_eval: float = -np.inf
        self.iteration = 1
        self.γ = self.reward_signals['extrinsic'].gamma
        self.mstep_iteration_num: int = self.hyperparameters.mstep_iteration_num

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.α_scale = self.hyperparameters.alpha_scale
        self.ε_kl = self.hyperparameters.kl_constraint
        self.α_max = self.hyperparameters.alpha_max

    def __update_params(self):
        # TODO: UNTESTED FOR OUR CASE
        # Copy policy parameters
        for target_param, param in zip(self.target_policy.actor.parameters(), self.policy.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    @property
    def critic(self):
        return self._critic

    @property
    def target_critic(self):
        return self._target_critic

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """parameters
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        n_obs = len(self.policy.behavior_spec.observation_specs)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        # Get value memories
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]
        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)

        state_batch = [ModelUtils.list_to_tensor(obs) for obs in ObsUtil.from_buffer(batch, n_obs)]
        action_batch = AgentAction.from_buffer(batch) # actions
        next_state_batch = [ModelUtils.list_to_tensor(obs) for obs in ObsUtil.from_buffer_next(batch, n_obs)]
        reward_batch = ModelUtils.list_to_tensor(batch[RewardSignalUtil.rewards_key('extrinsic')])
        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])

        ds = self.policy.behavior_spec.observation_specs # This is a list of observations
        da = self.policy.behavior_spec.action_spec.discrete_size
        
        # https://github.com/daisatojp/mpo

        # Policy Evaluation
        with torch.no_grad():
            run_out = self.target_policy.actor.get_stats(
                next_state_batch,
                action_batch,
                masks=act_masks,
                memories=memories,
                sequence_length=self.target_policy.sequence_length,
            )
            target_next_π_prob = run_out["log_probs"].discrete_tensor.exp() 
            expanded_next_states = next_state_batch
            for index, state in enumerate(expanded_next_states):
                state = state[:, None, :].expand(-1, da, -1).reshape(-1, ds[index].shape[0])

            expected_next_q, _ = self.target_critic.critic_pass(
                expanded_next_states,
                memories=value_memories,
                sequence_length=self.target_policy.sequence_length,
            )

            expected_next_q = torch.gather(expected_next_q['extrinsic'], 2, action_batch.discrete_tensor.unsqueeze(-1)).squeeze() 
            expected_next_q = expected_next_q * target_next_π_prob 
            expected_next_q = expected_next_q.sum(dim=-1)

            y = reward_batch + self.γ * expected_next_q
        self.critic_optimizer.zero_grad()
        t, _ = self.critic.critic_pass(
            state_batch,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )

        t = torch.gather(t['extrinsic'], 2, action_batch.discrete_tensor.unsqueeze(-1)).squeeze().sum(dim=-1) 
        loss = self.norm_loss_q(y, t)
        loss.backward()
        self.critic_optimizer.step()

        # E-Step of Policy Improvement
        with torch.no_grad():
            run_out = self.target_policy.actor.get_stats(
                state_batch,
                action_batch,
                masks=act_masks,
                memories=memories,
                sequence_length=self.target_policy.sequence_length,
            )
            target_π_prob = run_out["log_probs"].discrete_tensor.exp() 

            expanded_states = state_batch
            for index, state in enumerate(expanded_states):
                state = state[:, None, :].expand(-1, da, -1).reshape(-1, ds[index].shape[0])

            target_q, _ = self.target_critic.critic_pass(
                expanded_states,
                memories=value_memories,
                sequence_length=self.target_policy.sequence_length,
            )
            target_q = torch.gather(target_q['extrinsic'], 2, action_batch.discrete_tensor.unsqueeze(-1)).squeeze()

            target_π_prob_np = target_π_prob.cpu().numpy()
            target_q_np = target_q.cpu().numpy()

        # https://github.com/daisatojp/mpo
        def dual(η):
            max_q = np.max(target_q_np, 1)
            return η * self.ε_dual + np.mean(max_q) \
                + η * np.mean(np.log(np.sum(
                    target_π_prob_np * np.exp((target_q_np - max_q[:, None]) / η), axis=1)))

        bounds = [(1e-6, None)]
        res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
        self.η = res.x[0]

        qij = torch.softmax(target_q / self.η, dim=0)

        # M-Step of Policy Improvement
        for _ in range(self.mstep_iteration_num):
            run_out = self.policy.actor.get_stats(
                state_batch,
                action_batch,
                masks=act_masks,
                memories=memories,
                sequence_length=self.target_policy.sequence_length,
            ) 
            π_log_probs = run_out["log_probs"].discrete_tensor
            loss_p = torch.mean(
                qij * π_log_probs
            )

            π_p = π_log_probs.exp()

            kl = categorical_kl(p1=π_p, p2=target_π_prob)

            if np.isnan(kl.item()): 
                raise RuntimeError('kl is nan')

            self.α -= self.α_scale * (self.ε_kl - kl).detach().item()
            self.α = np.clip(self.α, 0.0, self.α_max)
            self.actor_optimizer.zero_grad()
            loss_l = -(loss_p + self.α * (self.ε_kl - kl))
            loss_l.backward()
            clip_grad_norm_(self.policy.actor.parameters(), 0.1)
            self.actor_optimizer.step()

        self.__update_params()

        update_stats = {
            # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
            # TODO: After PyTorch is default, change to something more correct.
            "Losses/loss_p": torch.abs(loss_p).item(),
            "Losses/Policy Loss": torch.abs(loss_l).item(),
        }

        return update_stats

    # TODO move module update into TorchOptimizer for reward_provider
    def get_modules(self):
        modules = {
            "Optimizer:critic_optimizer": self.critic_optimizer,
            "Optimizer:critic": self._critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

class QNetwork(nn.Module, Actor, Critic):
    MODEL_EXPORT_VERSION = 3

    def __init__(
        self,
        stream_names: List[str],
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        exploration_initial_eps: float = 1.0,
    ):
        self.exploration_rate = exploration_initial_eps
        nn.Module.__init__(self)
        self.network_body = BranchValueNetwork(
            stream_names,
            observation_specs,
            network_settings,
            outputs_per_stream=action_spec.discrete_branches,
        )

        # extra tensors for exporting to ONNX
        self.action_spec = action_spec
        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )
        self.is_continuous_int_deprecated = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.is_continuous())]), requires_grad=False
        )
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]), requires_grad=False
        )
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([self.action_spec.discrete_branches]), requires_grad=False
        )
        self.act_size_vector_deprecated = torch.nn.Parameter(
            torch.Tensor(
                [
                    self.action_spec.continuous_size
                    + sum(self.action_spec.discrete_branches)
                ]
            ),
            requires_grad=False,
        )
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.network_body.memory_size)]), requires_grad=False
        )

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        value_outputs, critic_mem_out = self.network_body(
            inputs, actions, memories=memories, sequence_length=sequence_length
        )
        return value_outputs, critic_mem_out

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        out_vals, memories = self.critic_pass(inputs, memories=memories, sequence_length=sequence_length)
        export_out = [self.version_number, self.memory_size_vector]

        disc_action_out = self.get_greedy_action(out_vals)
        deterministic_disc_action_out = self.get_random_action(out_vals)
        export_out += [
            disc_action_out,
            self.discrete_act_size_vector,
            deterministic_disc_action_out,
        ]

        return tuple(export_out)

    def get_random_action(self, inputs) -> torch.Tensor:
        action_out_list = []
        for branch_size in self.action_spec.discrete_branches:
            action_out_list.append(torch.randint(0, branch_size, (len(inputs), 1)))
        return action_out_list

    @staticmethod
    def get_greedy_action(q_values) -> torch.Tensor:
        all_q = torch.cat([val.unsqueeze(0) for val in q_values.values()])
        return torch.split(torch.argmax(all_q.sum(dim=0), dim=1, keepdim=True).squeeze(1), 1, dim=-1)

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        deterministic=False,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        run_out = {}
        if not deterministic and np.random.rand() < self.exploration_rate:
            action_out = self.get_random_action(inputs)
            agent_action_random_action_out = AgentAction(None, action_out)
            run_out["env_action"] = agent_action_random_action_out.to_action_tuple()
            return agent_action_random_action_out, run_out, torch.Tensor([])
        else:
            out_vals, _ = self.critic_pass(inputs, memories, sequence_length)
            action_out = self.get_greedy_action(out_vals)
            agent_action_greedy_action_out = AgentAction(None, action_out)
            run_out["env_action"] = agent_action_greedy_action_out.to_action_tuple()
            return agent_action_greedy_action_out, run_out, torch.Tensor([])
