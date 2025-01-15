import os
import numpy as np
import matplotlib.pyplot as plt
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env
from vmas import make_env
from vmas.simulator.environment import Wrapper

# 配置环境
scenario_name = "balance"
n_agents = 4
continuous_actions = True
max_steps = 200
num_vectorized_envs = 16
num_workers = 2
vmas_device = "cpu"


# 创建环境函数
def env_creator(config):
    return make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        **config["scenario_config"]
    )


# 注册环境
if not ray.is_initialized():
    ray.init()
register_env(scenario_name, lambda config: env_creator(config))


# 通用PPO配置模板
def get_ppo_config(algo_type="IPPO"):
    config = {
        "env": scenario_name,
        "framework": "torch",
        "seed": 0,
        "num_gpus": 0,
        "num_workers": num_workers,
        "num_envs_per_worker": num_vectorized_envs,
        "rollout_fragment_length": 200,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "lr": 5e-4,
        "gamma": 0.99,
        "clip_param": 0.2,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.01,
        "env_config": {
            "scenario_name": scenario_name,
            "device": vmas_device,
            "num_envs": num_vectorized_envs,
            "continuous_actions": continuous_actions,
            "max_steps": max_steps,
            "scenario_config": {
                "n_agents": n_agents,
            },
        },
    }

    if algo_type == "CPPO":
        config["simple_optimizer"] = True  # Centralized optimization
    elif algo_type == "MAPPO":
        config["multiagent"] = {
            "policies": {
                f"agent_{i}": (None, None, None, {}) for i in range(n_agents)
            },
            "policy_mapping_fn": lambda agent_id: f"agent_{agent_id}",
        }
    elif algo_type == "IPPO":
        config["multiagent"] = {
            "policies": {
                f"agent_{i}": (None, None, None, {}) for i in range(n_agents)
            },
            "policy_mapping_fn": lambda agent_id: f"agent_{agent_id}",
            "policies_to_train": [f"agent_{i}" for i in range(n_agents)],
        }

    return config


# 训练函数
def train_ppo(config, num_iterations=500):
    trainer = PPOTrainer(config=config)
    rewards = []

    for i in range(num_iterations):
        result = trainer.train()
        mean_reward = result["episode_reward_mean"]
        rewards.append(mean_reward)
        print(f"Iteration {i + 1}/{num_iterations}: Mean Reward = {mean_reward}")

    return rewards


# 测试函数
def test_policy(trainer, num_episodes=10):
    env_config = trainer.config["env_config"]
    test_env = env_creator(env_config)

    total_rewards = []
    for _ in range(num_episodes):
        obs = test_env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = {
                agent_id: trainer.compute_action(obs[agent_id], policy_id=f"agent_{agent_id}")
                for agent_id in obs
            }
            obs, reward, done, _ = test_env.step(action)
            episode_reward += sum(reward.values())

        total_rewards.append(episode_reward)

    return total_rewards


# 绘制奖励曲线
def plot_rewards(reward_dict):
    plt.figure(figsize=(10, 6))
    for algo, rewards in reward_dict.items():
        plt.plot(rewards, label=f"MAPPO Training Rewards")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Reward")
    plt.title("PPO Variants Training Rewards")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # algo_types = ["CPPO", "MAPPO", "IPPO"]
    all_rewards = {}
    # 测试每种算法
    algo = "CPPO"

    print(f"Starting training for {algo}...")
    ppo_config = get_ppo_config(algo_type=algo)
    rewards = train_ppo(ppo_config, num_iterations=300)
    all_rewards[algo] = rewards

    # 绘制所有算法的训练奖励曲线
    plot_rewards(all_rewards)

    print(f"Testing {algo} policy...")
    ppo_config = get_ppo_config(algo_type=algo)
    trainer = PPOTrainer(config=ppo_config)
    test_rewards = test_policy(trainer, num_episodes=10)
    print(f"{algo} Test Rewards: {test_rewards}")

    # 关闭Ray
    ray.shutdown()
