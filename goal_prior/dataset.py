import os
import pickle
from typing import Optional
from steve1.data.EpisodeStorage import EpisodeStorage
import numpy as np
import cv2
#from VPT.lib.tree_util import tree_map
from steve1.helpers import object_to_numpy, batch_recursive_objects
#from torch.utils.data import Dataset
from tqdm import tqdm
import random
#import matplotlib.pyplot as plt
from VPT.agent import AGENT_RESOLUTION, resize_image
from steve1.data.minecraft_dataset import MinecraftDataset

NONE_EMBED_OFFSET = 15

def env_obs_to_agent(frame):
    """
    Turn observation from MineRL environment into model's observation
    Returns torch tensors.
    """
    agent_input = resize_image(frame, AGENT_RESOLUTION)[None]
    return {"img": agent_input}


def get_episode_chunk(codebook, episode_chunk, min_btwn_goals, max_btwn_goals):
    """rewrite get_episode_chunk in minecraft_dataset.py
    remove mineclip embedding in the observation; 
    replace action with index of the most similar code in codebook to the goal 

    Args:
        episode_chunk (tuple): (episode_dirpath, start_timestep, end_timestep)
        min_btwn_goals (int): Minimum number of timesteps between goals.
        max_btwn_goals (int): Maximum number of timesteps between goals.
    """
    episode_dirpath, start_timestep, end_timestep = episode_chunk
    T = end_timestep - start_timestep
    episode = EpisodeStorage(episode_dirpath)

    # Get the goal embeddings
    embeds = episode.load_embeds_attn()

    frames = episode.load_frames(only_range=(start_timestep, end_timestep))
    total_timesteps = len(episode)

    # Choose goal timesteps
    goal_timesteps = []
    curr_timestep = 0
    while curr_timestep < total_timesteps - 1:
        curr_timestep += np.random.randint(min_btwn_goals, max_btwn_goals)
        if (total_timesteps - curr_timestep) < min_btwn_goals:
            curr_timestep = total_timesteps - 1
        goal_timesteps.append(curr_timestep)

    embeds_per_timestep = []
    # Tranlate into embeds per timestep
    cur_goal_timestep_idx = 0
    for t in range(total_timesteps):
        goal_timestep = goal_timesteps[cur_goal_timestep_idx]
        embed = embeds[goal_timestep]

        embeds_per_timestep.append(embed)
        if t == goal_timesteps[cur_goal_timestep_idx] + 1:
            # We've reached the timestep after the goal timestep, so move to the next goal
            cur_goal_timestep_idx += 1

    # compute embedding similarities to get action labels
    embeds_per_timestep = np.asarray(embeds_per_timestep)[start_timestep:end_timestep]
    embeds_per_timestep /= np.linalg.norm(embeds_per_timestep, axis=-1, keepdims=True)
    embeds_per_timestep = np.squeeze(embeds_per_timestep, axis=1) # (T, 512)
    codebook_T = np.squeeze(codebook / np.linalg.norm(codebook, axis=-1, keepdims=True), axis=1).transpose() #(512,N)
    sim_matrix = embeds_per_timestep @ codebook_T # (T,N)
    actions = np.argmax(sim_matrix, axis=1) # (T) int

    obs_list = []
    firsts_list = [True] + [False] * (T - 1)

    for t in range(start_timestep, end_timestep):
        frame = frames[t]
        obs = env_obs_to_agent(frame)
        obs_list.append(obs)

    obs_np = batch_recursive_objects(obs_list)
    actions_np = actions.reshape(T, 1)
    firsts_np = np.array(firsts_list, dtype=bool).reshape(T, 1)

    return obs_np, actions_np, firsts_np


def batch_if_numpy(xs):
    if isinstance(xs, np.ndarray):
        return np.array(xs)
    else:
        return xs


class MinecraftGoalPriorDataset(MinecraftDataset):
    def __init__(self, episode_dirnames, T, min_btwn_goals, max_btwn_goals,
                 p_uncond=None, limit=None, every_nth=None, codebook=None):
        self.codebook=np.asarray(codebook.codebook)
        super().__init__(episode_dirnames, T, min_btwn_goals, max_btwn_goals, p_uncond, limit, every_nth)

    def __getitem__(self, idx):
        obs_np, actions_np, firsts_np = \
            get_episode_chunk(self.codebook, 
                self.episode_chunks[idx], 
                self.min_btwn_goals, 
                self.max_btwn_goals)
        return obs_np, actions_np, firsts_np

    def collate_fn(self, batch):
        obs_np, actions_np, firsts_np = zip(*batch)
        obs = batch_recursive_objects(obs_np)
        actions = np.array(actions_np)
        firsts = np.array(firsts_np)