# train high-level policy with PPO for a task
import pickle
import argparse
import os
from datetime import datetime
import sys
import shutil

from hierarchical import PPO, PPOPolicy, POLICY_KWARGS
from sb3_vpt.logging import LoggingCallback
from tasks import make, get_specs
from steve1.utils.load_agent import make_agent
from goal_prior.load_prior import make_prior_model
from hierarchical import HighlevelWrapper
from gym3.types import Discrete, TensorType, Real
from hierarchical.load_codebook import KMeansCodebook
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ppo-highlevel", 
                        help="Name of the experiment, will be used to create a results directory.")
    parser.add_argument("--config", type=str, default="cobblestone",
                        help="Minedojo task to run. Should be exist in tasks/task_specs.yaml")
    parser.add_argument("--weights", type=str, default="downloads/rl-from-foundation-2x.weights",
                        help="Path to the initial model weights of VPT-2x.")
    parser.add_argument('--in_model', type=str, default='downloads/2x.model', help="Path to model parameters.")
    parser.add_argument('--in_weights', type=str, default='downloads/steve1/steve1.weights', help="Path to goal-conditioned model weights.")
    parser.add_argument("--cond_scale", type=float, default=7.0, help="Steve-1 cond scale.")
    parser.add_argument('--low_level_steps', type=int, default=100, help="Execution steps for low-level policy.")
    parser.add_argument("--discrete", action="store_true", help="Use discrete action space to index codebook.")
    parser.add_argument('--codebook', type=str, default='', help="path to codebook")
    parser.add_argument("--goal_prior", action="store_true", help="Use goal prior model to provide reward.")
    parser.add_argument("--initialize_with_prior", action="store_true", help="Initialize the policy with the goal prior model.")
    parser.add_argument('--prior_weights', type=str, default='results/goal_prior/weights/trained_with_script_best.weights', 
        help="Path to pre-trained goal prior model.")
    parser.add_argument("--kl_reward", type=float, default=1.0, help="Weight for KL divergence reward.")

    parser.add_argument("--load", type=str, default="",
                        help="Path to a zip filed to load from, saved by a previous run.")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Path to results dir.")
    parser.add_argument("--steps", type=int, default=100000, 
                        help="Total number of learner environement steps before learning stops.")
    parser.add_argument("--steps_per_iter", type=int, default=40,
                        help="Number of high-level steps per environment each iteration.")
    parser.add_argument("--batch_size", type=int, default=40,
                        help="Batch size for learning.")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Number of PPO epochs every iteration.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=.999,
                        help="Discount factor.")
    parser.add_argument("--cpu", action="store_true",
                        help="Use cpus over gpus.")
    args = parser.parse_args()

    _, task_specs, _ = get_specs(args.config)
    vars(args).update(**task_specs)
    #print(args)

    log_dir = os.path.join(args.results_dir, args.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "checkpoints"))

    # discrete codebook action space
    codebook = None
    if args.discrete:
        codebook = KMeansCodebook(path=args.codebook)
        POLICY_KWARGS["action_space"] = TensorType(shape=(1,), eltype=Discrete(codebook.N))
        #POLICY_KWARGS["pi_head_kwargs"] = {'temperature': 2.0}
        print('Codebook loaded:', args.codebook, 'size:', codebook.N)

    goal_conditioned_agent = make_agent(args.in_model, args.in_weights, args.cond_scale)
    goal_conditioned_agent.policy.eval()
    if args.goal_prior:
        prior_model = make_prior_model(args.in_model, args.prior_weights, POLICY_KWARGS['action_space'], strict=True)
        prior_model.policy.eval()
    else:
        prior_model = None
    env = make(args.config)
    env = HighlevelWrapper(env, goal_conditioned_agent, args.cond_scale, args.low_level_steps, 
        args.discrete, codebook, args.goal_prior, prior_model, args.kl_reward)


    if args.load:
        raise NotImplementedError
    else:
        agent_parameters = POLICY_KWARGS #pickle.load(open(args.model, "rb"))
        policy_kwargs = agent_parameters["policy_kwargs"] #agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["pi_head_kwargs"] #agent_parameters["model"]["args"]["pi_head_opts"]
        policy_action_space = agent_parameters["action_space"]

        model = PPO(
            PPOPolicy, 
            env, 
            n_steps=args.steps_per_iter,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device="cpu" if args.cpu else "cuda", 
            policy_kwargs=dict(
                policy_kwargs=policy_kwargs, 
                pi_head_kwargs=pi_head_kwargs,
                weights_path=args.weights,
                policy_action_space=policy_action_space
            ),
            tensorboard_log=os.path.join(log_dir, "tb"),
            learning_rate=args.lr,
            gamma=args.gamma,
            vf_coef=1,
            n_tasks=1,
            log_dir=log_dir,
            discrete=args.discrete,
            codebook=codebook,
            goal_prior=args.goal_prior,
            initialize_with_prior=args.initialize_with_prior,
            prior_path=args.prior_weights,
        )
    model.learn(
        args.steps, 
        callback=LoggingCallback(model, log_dir)
    )
