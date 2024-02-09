from typing import List, Tuple
import gym
from gym import spaces
import torch as th
import numpy as np
from copy import deepcopy

from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer

from sb3_vpt.buffer import VPTBuffer
import imageio, os

class PPO(RecurrentPPO):
    def __init__(self, *args, use_task_ids=False, n_tasks=1, log_dir=None, discrete=False, 
        codebook=None, goal_prior=False, use_spirl=False, initialize_with_prior=False, prior_path=None, **kwargs):
        #self.init_kl_coef = kl_coef
        #self.kl_coef = self.init_kl_coef
        #self.kl_decay = kl_decay
        self.use_task_ids = use_task_ids
        self.n_tasks = n_tasks
        assert n_tasks==1
        self.log_dir = log_dir
        self.discrete = discrete
        self.codebook = codebook
        self.goal_prior = goal_prior
        self.use_spirl = use_spirl
        self.initialize_with_prior = initialize_with_prior
        self.prior_path = prior_path
        super().__init__(*args, **kwargs)

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        return ["policy.optimizer"] + ["policy." + p for p in self.policy.get_param_keys()], []
    
    def set_task_id(self, task_id):
        self._last_task_id = np.full((self.n_envs,), task_id, dtype=np.int16)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            discrete=self.discrete,
            codebook=self.codebook,
            initialize_with_prior=self.initialize_with_prior,
            prior_path=self.prior_path,
            use_sde=self.use_sde,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        if not isinstance(self.policy, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        buffer_cls = VPTBuffer

        self._last_vpt_states = self.policy.initial_state(self.n_envs)  # num_blocks x batch x buffer, num_blocks x batch x buffer x hidden
        self._last_task_id = np.zeros((self.n_envs,), dtype=np.int16)

        hidden_state_buffer_shape = (
            self.n_steps, 
            self._last_vpt_states[1].shape[0],  # num_blocks
            self.n_envs,
            self._last_vpt_states[1].shape[3]  # hidden size
        )

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def reset(self):
        self._last_obs = self.env.reset()
        self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
        self._last_vpt_states = self.policy.initial_state(self.n_envs)
        self._last_task_id = np.zeros((self.n_envs,), dtype=np.int16)
        #self.kl_coef = self.init_kl_coef

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (VPTBuffer, RecurrentRolloutBuffer, RecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        vpt_states = deepcopy(self._last_vpt_states)

        save_rgb = True if self.num_timesteps % 500 == 0 else False
        rgb_list = []

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts).float().to(self.device)
                actions, values, log_probs, vpt_states, pd = self.policy.forward(obs_tensor, vpt_states, episode_starts, self._last_task_id)
                #print(actions, values)

            actions = actions.cpu().numpy()  # n_envs x 2
            #print(actions)
            if self.discrete:
                actions = actions[0]

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            #if isinstance(self.action_space, gym.spaces.Box):
            #    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            #print('action', clipped_actions)

            if self.goal_prior or self.use_spirl:
                assert env.num_envs==1
                rewards_kl = env.env_method('compute_kl_reward', pd[0])
                #print(rewards_kl)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            #print(new_obs, rewards, dones, infos)
            if self.goal_prior or self.use_spirl:
                rewards[0] += rewards_kl[0]
                #infos[0]['reward_kl'] = rewards_kl[0]

            if save_rgb:
                rgb_list.append(np.asarray(new_obs[0]).astype(np.uint8))
            #print('obs', new_obs)
            #print('actions', clipped_actions)

            # reset immediately if done
            if dones:
                infos[0]["terminal_observation"] = new_obs
                new_obs = env.reset()

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            '''
            # Update dones with task terminals after callbacks
            curr_task_id = np.array([
                info["subgoal"] if "subgoal" in info and self.use_task_ids else self._last_task_id[i]
                for i, info in enumerate(infos)
            ], dtype=np.int16)
            dones = np.bitwise_or(dones, curr_task_id != self._last_task_id)
            '''

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                vpt_states=self._last_vpt_states,
                task_id=self._last_task_id
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_vpt_states = vpt_states
            #self._last_task_id = curr_task_id

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones).float().to(self.device)
            ### VPT Changes Start: pass entire state to policy
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), vpt_states, episode_starts, self._last_task_id)
            ### VPT Changes End

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        if save_rgb:
            imageio.mimsave(os.path.join(self.log_dir, 'iter_step{}.gif'.format(self.num_timesteps)), rgb_list, duration=0.34)

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        pg_losses, value_losses, losses, bc_losses = [], [], [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                #print(rollout_data.observations.shape, actions.shape, rollout_data.vpt_states[0].shape, 
                #    rollout_data.episode_starts.shape, rollout_data.task_id.shape)
                values, log_prob = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.vpt_states,  # 4, 1, 128, 2048
                    rollout_data.episode_starts,
                    rollout_data.task_id
                )
                #print(values, log_prob)

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])
                #print(policy_loss, mask)

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                value_losses.append(value_loss.item())

                #kl_loss = th.mean(kl)
                #kl_losses.append(kl_loss.item())

                loss = policy_loss + self.vf_coef * value_loss # + self.kl_coef * kl_loss
                losses.append(loss.item())

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        #self.kl_coef *= self.kl_decay
        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        if len(bc_losses) > 0:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
        #self.logger.record("train/kl_loss", np.mean(kl_losses))
        #self.logger.record("train/kl_coef", self.kl_coef)
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/avg_loss", np.mean(losses))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

