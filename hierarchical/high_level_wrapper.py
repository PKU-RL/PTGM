#from gym import Wrapper
#import gym.spaces as spaces
import numpy as np
#from abc import ABC, abstractmethod
import torch
from gym.spaces import Box, Discrete

class HighlevelWrapper:
    def __init__(self, env, steve1_agent, cond_scale, low_level_steps=50, discrete=False, codebook=None, 
        use_prior=False, prior_model=None, kl_reward=1.0):
        self.env = env
        self.steve1_agent = steve1_agent
        self.cond_scale = cond_scale
        self.low_level_steps = low_level_steps
        self.observation_space = self.env.observation_space
        self.discrete = discrete
        self.codebook = codebook
        if not discrete:
            self.action_space = Box(low=-10., high=10., shape=(512,))
        else:
            self.action_space = Discrete(codebook.N)
        print('action space:', self.action_space)
        self.num_envs = 1
        self.metadata = None
        self.use_prior = use_prior
        self.prior_model = prior_model
        self.kl_reward = kl_reward

    def _prior_forward(self, obs):
        with torch.no_grad():
            obs_input = self.prior_model._env_obs_to_agent({'pov':obs})
            _, state, result = self.prior_model.policy.act(obs_input, self.prior_model._dummy_first, self.prior_model.hidden_state,
                return_pd=True)
            self.prior_model.hidden_state = state 
            self.prior_distribution = result['pd']
        #print('prior distribution: ', self.prior_distribution)

    def reset(self):
        self.steve1_agent.reset(cond_scale=self.cond_scale)
        self._last_obs = self.env.reset()
        self._last_action = 0
        self.total_steps = 0
        if self.use_prior:
            self.prior_model.reset()
            self._prior_forward(self._last_obs)
        return self._last_obs

    def step(self, action, save_rgb=False):
        if save_rgb:
            self.rgb_list = []
        if not self.discrete:
            goal_embed_action = action
        else:
            #print(action)
            goal_embed_action = self.codebook.get_code(action)
            #print(goal_embed_action)
        self.increase_steps = 0
        self._last_action = action
        cum_reward = 0.
        for i in range(self.low_level_steps):
            with torch.cuda.amp.autocast():
                minerl_action = self.steve1_agent.get_action({"pov": self._last_obs}, goal_embed_action)
            obs, reward, done, info = self.env.step(minerl_action, use_minerl_action=True)
            if self.use_prior:
                self._prior_forward(obs)
            if save_rgb:
                self.rgb_list.append(np.array(obs).astype(np.uint8))
            self._last_obs = obs 
            cum_reward += reward
            self.increase_steps += 1
            self.total_steps += 1 
            if done:
                return obs, cum_reward, done, info 
        return obs, cum_reward, done, info

    # KL reward to a prior model. should call before step()
    def compute_kl_reward(self, pd):
        #print(pd)
        reward_kl = -self.prior_model.policy.get_kl_of_action_dists(pd, self.prior_distribution).cpu().numpy().reshape(-1)[0]
        return self.kl_reward * reward_kl

    # inputs for mineagent
    def get_mineagent_obs(self):
        return self.env.minedojo_obs, self.env.clip_video_feats, self._last_action
