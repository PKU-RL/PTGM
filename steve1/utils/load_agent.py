import pickle
from steve1.MineRLConditionalAgent import MineRLConditionalAgent

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def make_agent(in_model, in_weights, cond_scale, strict=False):
    print(f'Loading agent with cond_scale {cond_scale}...')
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    #env = gym.make("MineRLBasaltFindCave-v0")
    # Make conditional agent
    #print(agent_policy_kwargs, agent_pi_head_kwargs)
    agent = MineRLConditionalAgent(env=None, device='cuda', policy_kwargs=agent_policy_kwargs,
                                   pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights, strict)
    agent.reset(cond_scale=cond_scale)
    #env.close()
    return agent
