import pickle
from hierarchical import HighlevelAgent

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def make_prior_model(in_model, in_weights, action_space, strict=True):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    agent = HighlevelAgent(device='cuda', action_space=action_space, 
        policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights, strict)
    agent.reset()
    return agent
