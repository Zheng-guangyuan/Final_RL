import os
from typing import Dict, Optional

import numpy as np
import ray
from ray import tune
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env

from vmas import make_env
from vmas.simulator.environment import Wrapper

# Scenario specific variables
scenario_name = "balance"
n_agents = 4  # Number of agents in the environment

# Common variables
continuous_actions = True
max_steps = 200
num_vectorized_envs = 96
num_workers = 5
vmas_device = "cpu"  # or cuda


def env_creator(config: Dict):
    """
    Create a centralized version of the VMAS environment.
    """
    base_env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        **config["scenario_config"],
    ).vector_env

    class CentralizedEnv:
        def __init__(self, env):
            self.env = env
            self.n_agents = config["scenario_config"]["n_agents"]

        def reset(self):
            obs = self.env.reset()
            centralized_obs = np.concatenate([obs[agent_id] for agent_id in obs], axis=0)
            return {"centralized": centralized_obs}

        def step(self, action_dict):
            # Split centralized action into individual actions
            actions = np.split(action_dict["centralized"], self.n_agents)
            obs, rewards, dones, infos = self.env.step(actions)

            centralized_obs = np.concatenate([obs[agent_id] for agent_id in obs], axis=0)
            centralized_reward = np.mean([rewards[agent_id] for agent_id in rewards])
            dones["__all__"] = all(dones.values())
            return {"centralized": centralized_obs}, {"centralized": centralized_reward}, dones, infos

    return CentralizedEnv(base_env)


if not ray.is_initialized():
    ray.init()
    print("Ray initialized!")

register_env(scenario_name, lambda config: env_creator(config))

def train():
    RLLIB_NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    num_gpus = 0.001 if RLLIB_NUM_GPUS > 0 else 0  # Driver GPU
    num_gpus_per_worker = (
        (RLLIB_NUM_GPUS - num_gpus) / (num_workers + 1) if vmas_device == "cuda" else 0
    )

    tune.run(
        PPOTrainer,
        stop={"training_iteration": 1000},
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        config={
            "seed": 0,
            "framework": "torch",
            "env": scenario_name,
            "kl_coeff": 0.01,
            "kl_target": 0.01,
            "lambda": 0.9,
            "clip_param": 0.2,
            "vf_loss_coeff": 1,
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0,
            "train_batch_size": 60000,
            "rollout_fragment_length": 125,
            "sgd_minibatch_size": 4096,
            "num_sgd_iter": 40,
            "num_gpus": num_gpus,
            "num_workers": num_workers,
            "num_gpus_per_worker": num_gpus_per_worker,
            "num_envs_per_worker": num_vectorized_envs,
            "lr": 5e-5,
            "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "batch_mode": "truncate_episodes",
            "multiagent": {
                "policies": {"shared_policy": (None, None, None, {})},
                "policy_mapping_fn": lambda agent_id: "shared_policy",
            },
            "env_config": {
                "device": vmas_device,
                "num_envs": num_vectorized_envs,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_steps,
                "scenario_config": {
                    "n_agents": n_agents,
                },
            },
            "evaluation_interval": 5,
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {
                    "num_envs": 1,
                },
            },
        },
    )


if __name__ == "__main__":
    train()
