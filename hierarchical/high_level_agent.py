import numpy as np
import torch as th
import inspect
import cv2
from gym3.types import DictType
from gym import spaces

from hierarchical import MinecraftAgentPolicy
from VPT.agent import default_device_type, set_default_torch_device

# Hardcoded settings
AGENT_RESOLUTION = (128, 128)


def resize_image(img, target_resolution):
    #print(img.shape, target_resolution)
    # For your sanity, do not resize with any function than INTER_LINEAR
    # Only resize if needed (target_resolution is flipped for opencv)
    if img.shape[:2] != target_resolution[::-1]:
        img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img


class HighlevelAgent:
    def __init__(self, device=None, action_space=None, policy_kwargs=None, pi_head_kwargs=None):
        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)

        agent_kwargs = dict(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=action_space)
        self.action_space = action_space

        self.policy = MinecraftAgentPolicy(**agent_kwargs).to(device)
        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)

    def load_weights(self, path, strict=False):
        """Load model weights from a path, and reset hidden state"""
        state_dict = th.load(path, map_location=self.device)
        state_dict = {k.replace("policy.", ""): v for k, v in state_dict.items()}
        self.policy.load_state_dict(state_dict, strict=strict)
        self.reset()

    def reset(self):
        """Reset agent to initial state (i.e., reset hidden state)"""
        self.hidden_state = self.policy.initial_state(1)

    def _env_obs_to_agent(self, minerl_obs):
        """
        Turn observation from MineRL environment into model's observation

        Returns torch tensors.
        """
        agent_input = resize_image(minerl_obs["pov"], AGENT_RESOLUTION)[None]
        agent_input = {"img": th.from_numpy(agent_input).to(self.device)}
        return agent_input

    def get_action(self, minerl_obs):
        """
        Get agent's action for given MineRL observation.

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        agent_input = self._env_obs_to_agent(minerl_obs)
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        agent_action, self.hidden_state, _ = self.policy.act(
            agent_input, self._dummy_first, self.hidden_state,
            stochastic=True
        )
        agent_action = agent_action.cpu().numpy()
        return agent_action


# optimizer for supervised training on dataset
def configure_optimizers(model, weight_decay, learning_rate):

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (th.nn.Linear, )
    blacklist_weight_modules = (th.nn.LayerNorm, th.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            else:
                decay.add(fpn)

    # If a parameter is in both decay and no_decay, remove it from decay.
    decay = decay - no_decay

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )
    
    # Print the keys that are in each set in a comma-separated list.
    # print(f"decay keys: {', '.join(sorted(list(decay)))}")
    # print(f"no decay keys: {', '.join(sorted(list(no_decay)))}")

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    use_fused = 'fused' in inspect.signature(th.optim.AdamW).parameters
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()

    # Make sure that all parameters are CUDA and floating point Tensor.
    # This is required by fused optimizer.
    # for group in optim_groups:
    #     for p in group["params"]:
    #         if not p.is_cuda:
    #             print(f"WARNING: parameter {p} is not CUDA tensor. Fixing for AdamW with fused optimizer.")
    #             p.data = p.data.cuda()
    #         if not p.is_floating_point():
    #             print(f"WARNING: parameter {p} is not floating point type. Fixing for AdamW with fused optimizer.")
    #             p.data = p.data.float()
    optimizer = th.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)

    return optimizer
