# wrapper for some human-defined tasks which need success rewards computed from additional env information
from typing import Dict
from gym import Wrapper
from abc import ABC, abstractstaticmethod


class OpentaskWrapper(Wrapper, ABC):
    def __init__(self, env, task):
        super().__init__(env)
        self.task = task
        assert task in ["tower", "place_a_block"]

    def reset(self):
        obs = super().reset()
        if self.task == "tower":
            self.init_h = obs["location_stats"]["pos"][1]
            #print("Task: build a tower. Initial height:", self.init_h)
            self.max_dh = 0
        elif self.task == "place_a_block":
            self.mainhand_block_num = obs["inventory"]["quantity"][0]
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.task=="tower":
            dh = int(obs["location_stats"]["pos"][1] - self.init_h)
            if dh > self.max_dh:
                reward += (dh-self.max_dh) # success reward for height-increasing
                self.max_dh = dh
                info["max_tower_height"] = self.max_dh
            else:
                info["max_tower_height"] = self.max_dh

        elif self.task == "place_a_block":
            mainhand_block_num = obs["inventory"]["quantity"][0]
            info["success"] = False
            if mainhand_block_num < self.mainhand_block_num:
                info["success"] = True
                reward += 1.0
                done=True
            self.mainhand_block_num = mainhand_block_num
        return obs, reward, done, info
