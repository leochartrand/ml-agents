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
    episode_rerun_num: int = 3
    mstep_iteration_num: int = 5 # Number of iterations for M-Step
    evaluate_episode_maxstep: int = 200 # Maximum evaluate steps of an episode

    # The following might not be needed
    sample_episode_num: int = 30
    sample_episode_max_steps:int = 200
    sample_action_num: int = 64

    # The rest of the parameters are what's classic for ML Agents.
    buffer_size: int = 2000
    critic_learning_rate: float = 0.001
    actor_learning_rate: float = 0.001
    num_epoch: int = 3
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR

    gamma: float = 0.99
    shared_critic: bool = False


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
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
        

        self.hyperparameters: MPOSettings = cast(
            MPOSettings, trainer_settings.hyperparameters
        )

        print("policy.behavior_spec.observation_specs: ", policy.behavior_spec.observation_specs)

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

        self.decay_actor_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.actor_learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )

        self.decay_critic_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.critic_learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )

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

        self.stream_names = list(self.reward_signals.keys())
    
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
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Get decayed parameters
        decay_actor_lr = self.decay_actor_learning_rate.get_value(self.policy.get_current_step())
        decay_critic_lr = self.decay_critic_learning_rate.get_value(self.policy.get_current_step())
        returns = {}
        old_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        # act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        # actions = AgentAction.from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        
        state_batch = ObsUtil.from_buffer(batch, n_obs) # current_obs
        action_batch = AgentAction.from_buffer(batch) # actions
        next_state_batch = ObsUtil.from_buffer_next(batch, n_obs) # current_obs
        reward_batch = 0 # TODO: reward

        K = len(action_batch.discrete_list) # Number of states, same len for actions than for states. Just simpler to use the action list here.
        # N = # TODO
        ds = self.policy.behavior_spec.observation_specs # This is a list of observations
        da = self.policy.behavior_spec.action_spec.discrete_size
        
        # Policy Evaluation
        # [2] 3 Policy Evaluation (Step 1)
        
        loss_q, q = self.__update_critic_td(
            state_batch = state_batch, 
            action_batch=action_batch, 
            next_state_batch=next_state_batch, 
            reward_batch=reward_batch, 
            ds=ds, 
            da=da
        )
        
        # E-Step of Policy Improvement
        # [2] 4.1 Finding action weights (Step 2)

        # TODO: Replace
        target_q_np = np.array([])
        b_prob_np = np.array([])
        
        # [2] 4.1 Finding action weights (Step 2)
        #   Using an exponential transformation of the Q-values
        def dual(self, η): # target_q_np & b_prob_np dims: (K, da)
            # TODO: UNTESTED FOR OUR CASE
            max_q = np.max(target_q_np, 1)
            return η * self.ε_dual + np.mean(max_q) \
                + η * np.mean(np.log(np.sum(
                    b_prob_np * np.exp((target_q_np - max_q[:, None]) / η), axis=1)))
        
        bounds = [(1e-6, None)]
        res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
        self.η = res.x[0]

        qij = torch.softmax(target_q / self.η, dim=0) # (N, K) or (da, K)

        # M-Step of Policy Improvement
        # [2] 4.2 Fitting an improved policy (Step 3)
        for _ in range(self.mstep_iteration_num):
            π_p = self.policy.actor.forward(state_batch)
            π = Categorical(probs=π_p)
            # ...
        
        
        
        
        # if len(memories) > 0:
        #     memories = torch.stack(memories).unsqueeze(0)

        # # Get value memories
        # value_memories = [
        #     ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
        #     for i in range(
        #         0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
        #     )
        # ]
        # if len(value_memories) > 0:
        #     value_memories = torch.stack(value_memories).unsqueeze(0)

        # run_out = self.policy.actor.get_stats(
        #     current_obs,
        #     actions,
        #     masks=act_masks,
        #     memories=memories,
        #     sequence_length=self.policy.sequence_length,
        # )

        # log_probs = run_out["log_probs"]
        # entropy = run_out["entropy"]

        # # Etape 1 : Policy Evaluation
        # values, _ = self.critic.critic_pass(
        #     current_obs,
        #     memories=value_memories,
        #     sequence_length=self.policy.sequence_length,
        # )

        # old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        # log_probs = log_probs.flatten()
        # loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        # value_loss = ModelUtils.trust_region_value_loss(
        #     values, old_values, returns, decay_eps, loss_masks
        # )
        # policy_loss = ModelUtils.trust_region_policy_loss(
        #     ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
        #     log_probs,
        #     old_log_probs,
        #     loss_masks,
        #     decay_eps,
        # )
        # loss = (
        #     policy_loss
        #     + 0.5 * value_loss
        #     - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
        # )

        # # Set optimizer learning rate
        # ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        # self.optimizer.zero_grad()
        # loss.backward()

        # self.optimizer.step()
        # update_stats = {
        #     # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
        #     # TODO: After PyTorch is default, change to something more correct.
        #     "Losses/Policy Loss": torch.abs(policy_loss).item(),
        #     "Losses/Value Loss": value_loss.item(),
        #     "Policy/Learning Rate": decay_lr,
        #     "Policy/Epsilon": decay_eps,
        #     "Policy/Beta": decay_bet,
        # }

        # return update_stats

    # TODO move module update into TorchOptimizer for reward_provider
    def get_modules(self):
        modules = {
            "Optimizer:critic_optimizer": self.critic_optimizer,
            "Optimizer:critic": self._critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

    def __update_critic_td(self,
                           state_batch,
                           action_batch,
                           next_state_batch,
                           reward_batch,
                           ds,
                           da):
        """
        :param state_batch: (B, ds)
        :param action_batch: (B, da) or (B,)
        :param next_state_batch: (B, ds)
        :param reward_batch: (B,)
        :return:
        """
        A_eye = torch.eye(da).to(default_device())

        # TODO: NOT TESTED FOR OUR CASE
        B = state_batch.size(0)
        with torch.no_grad():
            r = reward_batch  # (B,)
            π_p = self.target_policy.actor.forward(next_state_batch)  # (B, da)
            π = Categorical(probs=π_p)  # (B,)
            π_prob = π.expand((da, B)).log_prob(
                torch.arange(da)[..., None].expand(-1, B).to(self.device)  # (da, B)
            ).exp().transpose(0, 1)  # (B, da)
            sampled_next_actions = A_eye[None, ...].expand(B, -1, -1)  # (B, da, da)
            expanded_next_states = next_state_batch[:, None, :].expand(-1, da, -1)  # (B, da, ds)
            expected_next_q = (
                self.target_critic.forward(
                    expanded_next_states.reshape(-1, ds),  # (B * da, ds)
                    sampled_next_actions.reshape(-1, da)  # (B * da, da)
                ).reshape(B, da) * π_prob  # (B, da)
            ).sum(dim=-1)  # (B,)
            y = r + self.γ * expected_next_q
        self.critic_optimizer.zero_grad()

        t = self.critic(
            state_batch,
            A_eye[action_batch.long()]
        ).squeeze(-1)  # (B,)
        
        loss = self.norm_loss_q(y, t)
        loss.backward()
        self.critic_optimizer.step()
        return loss, y



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
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        value_outputs, critic_mem_out = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
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
        logger.debug('MAB: forward')

        out_vals, memories = self.critic_pass(inputs, memories, sequence_length)
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
