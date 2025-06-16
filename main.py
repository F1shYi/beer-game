from utils import (
    train_agent,
    test_agent,
    plot_training_results,
    save_test_scores,
    save_test_fill_rates,
    plot_test_inventories,
    plot_test_orders,
    plot_test_satisfied_demands,
    plot_different_envs
)
from environment import EnvBase, EnvSimple, EnvSeasonal, EnvComplex
from agent import AgentBase, AgentDQN, AgentPPO, AgentSAC
from collections import defaultdict
import torch
import json
import argparse

def _get_agent(agent_name: str):
    if agent_name == "DQN":
        return AgentDQN(3, 20, 0)
    elif agent_name == "PPO":
        return AgentPPO(3, 20, 0)
    elif agent_name == "SAC":
        return AgentSAC(3, 20, 0)
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")

def _get_env(env_name: str):
    if env_name == "EnvSimple":
        return EnvSimple(3, [10, 9, 8], 0.5, 2, 100)
    elif env_name == "EnvSimpleL":
        return EnvSimple(3, [10, 9, 8], 0.1, 2, 100)
    elif env_name == "EnvSeasonal":
        return EnvSeasonal(3, [10, 9, 8], 0.5, 2, 100)
    elif env_name == "EnvSeasonalL":
        return EnvSeasonal(3, [10, 9, 8], 0.1, 2, 100)
    elif env_name == "EnvComplex":
        return EnvComplex(3, [10, 9, 8], 0.5, 2, 100)
    elif env_name == "EnvComplexL":
        return EnvComplex(3, [10, 9, 8], 0.1, 2, 100)
    else:
        raise ValueError(f"Unknown env name: {env_name}")


def train(env_name, agent_name, lower_cost: bool = False):
    
    lower_cost=lower_cost
    
    env = _get_env(env_name)
    agent = _get_agent(agent_name)
    print(f"Training {agent_name} on {env_name}")
    print(f"Lower cost mode: {lower_cost}")
    scores = train_agent(env, agent, num_episodes=1500, max_steps=100,lower_cost=lower_cost)
    log_fpath = "models/" + f"Agent{agent_name}_{env_name}.json"
    with open(log_fpath, "w") as f:
        json.dump(scores, f)
   
def train_parser():
    """
    创建一个命令行解析器，用于指定训练的环境和智能体。
    """
    parser = argparse.ArgumentParser(description="Train an agent on a specified environment.")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["EnvSimple", "EnvSimpleL", "EnvSeasonal", "EnvSeasonalL", "EnvComplex", "EnvComplexL"],
        help="Env name"
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["DQN", "PPO", "SAC"],
        help="Agent name"
    )
    return parser



if __name__ == "__main__":
    parser = train_parser()
    args = parser.parse_args()
    train(args.env, args.agent, lower_cost=args.env.endswith("L"))
   
