from gym import Wrapper
import torch as th


class ClipWrapper(Wrapper):
    def __init__(self, env, clip, prompts=None, dense_reward=.01, clip_target=23, clip_min=21, smoothing=50, **kwargs):
        super().__init__(env)
        self.clip = clip

        assert prompts is not None
        self.prompt = prompts
        self.dense_reward = dense_reward
        self.smoothing = smoothing
        self.clip_target = th.tensor(clip_target)
        self.clip_min = th.tensor(clip_min)
        
        self.buffer = None
        self._clip_state = None, None
        self.last_score = 0

    def reset(self, **kwargs):
        self._clip_state = None, self._clip_state[1]
        self.buffer = None
        self.last_score = 0
        obs = self.env.reset(**kwargs)
        self.minedojo_obs = obs # save all the obs info in MineDojo
        if len(self.prompt) > 0:
            _, _, self.clip_video_feats = self.clip.get_logits(obs, self.prompt, self._clip_state)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.minedojo_obs = obs # save all the obs info in MineDojo

        if len(self.prompt) > 0:
            logits, self._clip_state, self.clip_video_feats = self.clip.get_logits(obs, self.prompt, self._clip_state)
            logits = logits.detach().cpu()

            self.buffer = self._insert_buffer(self.buffer, logits[:1])
            score = self._get_score()

            if score > self.last_score:
                reward += self.dense_reward * score
                self.last_score = score

        return obs, reward, done, info 

    def _get_score(self):
        return (max(
            th.mean(self.buffer) - self.clip_min,
            0
        ) / (self.clip_target - self.clip_min)).item()

    def _insert_buffer(self, buffer, logits):
        if buffer is None:
            buffer = logits.unsqueeze(0)
        elif buffer.shape[0] < self.smoothing:
            buffer = th.cat([buffer, logits.unsqueeze(0)], dim=0)
        else:
            buffer = th.cat([buffer[1:], logits.unsqueeze(0)], dim=0)
        return buffer
