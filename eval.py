from utils import (
    test_agent,
    plot_training_results,
    save_test_scores,
    save_test_fill_rates,
    plot_test_inventories,
    plot_test_orders,
    plot_test_satisfied_demands,
    plot_different_envs,
    plot_test_fill_rates
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


def test(lower_cost=False):
    agent_names = ["DQN", "SAC", "PPO"]
    env_names = ["EnvSimple", "EnvSeasonal", "EnvComplex"]
    if lower_cost:
        env_names = [env_name + "L" for env_name in env_names]
    
    test_scores = defaultdict(lambda: defaultdict(list))
    test_inventories = defaultdict(lambda: defaultdict(list))
    test_orders = defaultdict(lambda: defaultdict(list))
    test_demands = defaultdict(lambda: defaultdict(list))
    test_satisfied_demands = defaultdict(lambda: defaultdict(list))
    env_demands = defaultdict(list)
    for env_name in env_names:
        env = _get_env(env_name)
        env_demands[env_name] = env.get_demand()
        for agent_name in agent_names:
            agent = _get_agent(agent_name)
            print(f"Testing {agent_name} on {env_name}")
            agent.load(f"models/Agent{agent_name}_{env_name}.pth")
            (   scores,
                inventory_history,
                orders_history,
                demand_history,
                satisfied_demand_history,
            ) = test_agent(env, agent, 2000)
            test_scores[env_name][agent_name] = scores
            test_inventories[env_name][agent_name] = inventory_history
            test_orders[env_name][agent_name] = orders_history
            test_demands[env_name][agent_name] = demand_history
            test_satisfied_demands[env_name][agent_name] = satisfied_demand_history
    save_test_scores(test_scores)
    save_test_fill_rates(test_demands, test_satisfied_demands)
    plot_test_inventories(test_inventories)
    plot_test_orders(test_orders)
    plot_test_satisfied_demands(test_satisfied_demands, env_demands)
    plot_test_fill_rates(test_satisfied_demands, test_demands)
    
def plot_envs():
    env_simple = _get_env("EnvSimple")
    env_seasonal = _get_env("EnvSeasonal")
    env_complex = _get_env("EnvComplex")

    plot_different_envs(env_simple, env_seasonal, env_complex)

    
def plot_train_curves(lower_cost=False):
    agent_names = ["DQN", "SAC", "PPO"]
    env_names = ["EnvSimple", "EnvSeasonal", "EnvComplex"]
    if lower_cost:
        env_names = [env_name + "L" for env_name in env_names]
    
    training_scores = defaultdict(lambda: defaultdict(list))
    for agent_name in agent_names:
        for env_name in env_names:
            json_fpath = f"models/Agent{agent_name}_{env_name}.json"
            with open(json_fpath, "r") as f:
                data = json.load(f)
            training_scores[env_name][agent_name] = data
    if lower_cost:
        plot_training_results(training_scores, ymin=-100, ymax=1600)
        return
    plot_training_results(training_scores)


if __name__ == "__main__":
    plot_train_curves(True)
    test(True)
    
