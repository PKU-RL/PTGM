from typing import Dict
from omegaconf import OmegaConf
from minedojo.tasks import MetaTaskBase, _meta_task_make, _parse_inventory_dict, ALL_TASKS_SPECS
from minedojo.sim import MineDojoSim

from tasks.minedojo.wrappers import *
from tasks.base import OpentaskWrapper
import numpy as np


def _get_minedojo_specs(task_id, task_specs, sim_specs):
    if task_id in ALL_TASKS_SPECS:
        minedojo_specs = ALL_TASKS_SPECS[task_id]
        if OmegaConf.is_config(minedojo_specs):
            minedojo_specs = OmegaConf.to_container(minedojo_specs)
        minedojo_specs.pop("prompt", None)
        meta_task_cls = minedojo_specs.pop("__cls__")
    else:
        minedojo_specs = dict()
        meta_task_cls = task_id

    minedojo_specs.update(dict(
        image_size=(160, 256), 
        fast_reset=False, 
        event_level_control=False,
        use_voxel=False,
        use_lidar=False
    ))

    # If using blocks condition, activate voxels
    if ("terminal_specs" in task_specs and \
            ("all" in task_specs["terminal_specs"] and any(x == "blocks" for x in task_specs["terminal_specs"]["all"]) or \
            "any" in task_specs["terminal_specs"] and any(x == "blocks" for x in task_specs["terminal_specs"]["any"]))) or \
        ("success_specs" in task_specs and \
            ("all" in task_specs["success_specs"] and any(x == "blocks" for x in task_specs["success_specs"]["all"]) or \
            "any" in task_specs["success_specs"] and any(x == "blocks" for x in task_specs["success_specs"]["any"]))):
        minedojo_specs["use_voxel"] = True
        minedojo_specs["voxel_size"] = dict(xmin=-3, ymin=-1, zmin=-3, xmax=3, ymax=1, zmax=3)

    minedojo_specs.update(**sim_specs)

    # If using lidar, set the lidar range
    if ("use_lidar" in sim_specs and sim_specs["use_lidar"]):
        minedojo_specs["lidar_rays"] = [
            (np.pi * pitch / 180, np.pi * yaw / 180, 99)
            for pitch in np.arange(-30, 30, 5)
            for yaw in np.arange(-45, 45, 5)]

    if "initial_inventory" in minedojo_specs:
        minedojo_specs["initial_inventory"] = _parse_inventory_dict(minedojo_specs["initial_inventory"])

    return meta_task_cls, minedojo_specs


def _add_wrappers(
    env: MetaTaskBase, 
    task_id: str, 
    reward_specs: Dict = None,
    success_specs: Dict = None,
    terminal_specs: Dict = None,
    clip_specs: Dict = None,
    techtree_specs: Dict = None,
    fast_reset: int = None,
    log_dir: str = None,
    freeze_equipped: bool = False,
    opentask_specs: Dict = None,
    **kwargs
):
    if opentask_specs: # added 9/15
        env = OpentaskWrapper(env, **opentask_specs)
    if reward_specs:
        env = MinedojoRewardWrapper(env, **reward_specs)
    if success_specs:
        env = MinedojoSuccessWrapper(env, **success_specs)
    if terminal_specs is None:
        terminal_specs = dict(max_steps=500, on_death=True)
    env = MinedojoTerminalWrapper(env, **terminal_specs)

    # Add reward shaping wrapper
    if clip_specs is not None:
        clip_reward = MinedojoClipReward()
        env = ClipWrapper(env, clip_reward, **clip_specs)

    if techtree_specs is not None:
        env = MinedojoTechTreeWrapper(env, log_dir=log_dir, **techtree_specs)

    # Add VPT wrapper
    env = MinedojoVPTWrapper(env, freeze_equipped=freeze_equipped)

    # If we don't care about start position, use fast reset to speed training and prevent memory leaks
    if fast_reset is not None:
        wrapped = env
        while hasattr(wrapped, "env"):
            if isinstance(wrapped.env, MineDojoSim):
                wrapped.env = MinedojoSemifastResetWrapper(
                    wrapped.env,
                    reset_freq=fast_reset,
                    random_teleport_range=200
                )
                break
            wrapped = wrapped.env

    return env


def make_minedojo(task_id: str, task_specs, sim_specs):

    # Get minedojo specs
    meta_task_cls, minedojo_specs = _get_minedojo_specs(task_id, task_specs, sim_specs)
    print('meta_task_cls:', meta_task_cls, 'task_specs:', task_specs, 'minedojo_env_specs:', minedojo_specs)

    # Make minedojo env
    env = _meta_task_make(meta_task_cls, **minedojo_specs)

    # Add our wrappers
    env = _add_wrappers(env, task_id, **task_specs)

    return env
