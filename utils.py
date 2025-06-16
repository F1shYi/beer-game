from environment import EnvBase, EnvSimple, EnvSeasonal, EnvComplex
from agent import AgentBase
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def train_agent(
    env: EnvBase,
    agent: AgentBase,
    num_episodes=2000,
    max_steps=100,
    lower_cost=False,
):
    """
    训练智能体

    :param env: 环境 (来自 environment.py)
    :param agent: 智能体 (继承自 agent.py 中的 BaseAgent)
    :param num_episodes: 训练的episodes数量
    :param max_steps: 每个episode的最大步数
    :return: 所有episode的奖励
    """
    scores = []  # 每个episode的总奖励

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0

        for _ in range(max_steps):
            # 对特定企业采取动作，其他企业随机决策
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    # 使用智能体策略
                    firm_state = state[firm_id].reshape(1, -1)

                    action = agent.act(firm_state)
                    actions[firm_id] = action
                else:
                    # 对其他企业采取随机策略
                    actions[firm_id] = np.random.randint(1, 21)

            # 执行动作

            next_state, rewards, done = env.step(actions)

            # 该企业的奖励
            reward = rewards[agent.firm_id][0]

            # 保存经验并学习
            agent.step(
                state[agent.firm_id].reshape(1, -1),
                actions[agent.firm_id],
                reward,
                next_state[agent.firm_id].reshape(1, -1),
                done,
            )

            # 更新状态和奖励
            state = next_state
            score += reward

            if done:
                break

        # 记录分数
        scores.append(score)

        # 输出进度
        if i_episode % 100 == 0:
            print(
                f"Episode {i_episode}/{num_episodes} | Average Score: {np.mean(scores[-100:]):.2f}"
            )

    # 训练结束后保存最终模型
    l = "L" if lower_cost else ""
    agent.save(f"models/{agent.__class__.__name__}_{env.__class__.__name__}{l}.pth")

    return scores


def test_agent(env: EnvBase, agent: AgentBase, num_episodes=10):

    scores = []
    inventory_history = []
    orders_history = []
    demand_history = []
    satisfied_demand_history = []

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0
        episode_inventory = []
        episode_orders = []
        episode_demand = []
        episode_satisfied_demand = []

        for t in range(env.max_steps):
            # 对特定企业采取动作，其他企业随机决策
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    # 使用智能体策略，不使用探索
                    firm_state = state[firm_id].reshape(1, -1)
                    action = agent.act(firm_state)
                    actions[firm_id] = action
                else:
                    # 对其他企业采取随机策略
                    actions[firm_id] = np.random.randint(1, 21)

            # 执行动作
            next_state, rewards, done = env.step(actions)

            # 记录关键指标
            episode_inventory.append(env.inventory[agent.firm_id][0])
            episode_orders.append(actions[agent.firm_id][0])
            episode_demand.append(env.demand[agent.firm_id][0])
            episode_satisfied_demand.append(env.satisfied_demand[agent.firm_id][0])

            # 该企业的奖励
            reward = rewards[agent.firm_id][0]
            score += reward

            # 更新状态
            state = next_state

            if done:
                break

        # 记录分数和历史数据
        scores.append(score)
        inventory_history.append(episode_inventory)
        orders_history.append(episode_orders)
        demand_history.append(episode_demand)
        satisfied_demand_history.append(episode_satisfied_demand)

        print(f"Test Episode {i_episode}/{num_episodes} | Score: {score:.2f}")

    return (
        scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )




def plot_training_results(all_agents_scores, window_size=100, file_name="training_rewards.png", ymin=-700, ymax=1100):
    """
    绘制多个环境中多个Agent的训练结果

    :param all_agents_scores: 字典，key为环境名称，value为该环境中各Agent的scores字典
    :param window_size: 滑动平均窗口大小，默认为100
    """
    sns.set_theme(style="whitegrid")
    sns.set_theme(font="serif")
    plt.rcParams["font.family"] = "serif"

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(33, 8))

    # 计算移动平均
    def moving_average(data, window_size):
        return [
            np.mean(data[max(0, i - window_size) : i + 1]) for i in range(len(data))
        ]

    # 获取所有Agent的名称
    all_agent_names = set()
    for agents_scores in all_agents_scores.values():
        all_agent_names.update(agents_scores.keys())
    all_agent_names = sorted(all_agent_names)

    # 为每个Agent分配不同的颜色
    colors = sns.color_palette("husl", len(all_agent_names))
    agent_color_map = {
        agent_name: color for agent_name, color in zip(all_agent_names, colors)
    }

    for env_idx, (env_name, agents_scores) in enumerate(all_agents_scores.items()):
        ax = axs[env_idx]

        for agent_name, scores in agents_scores.items():
            color = agent_color_map[agent_name]
            episodes = np.arange(len(scores))
            avg_scores = moving_average(scores, window_size)

            # 绘制原始scores（浅色）
            sns.lineplot(x=episodes, y=scores, ax=ax, color=color, alpha=0.3)

            # 绘制滑动平均scores（深色，线条更粗）
            sns.lineplot(x=episodes, y=avg_scores, ax=ax, color=color, linewidth=2)

        ax.set_title(f"{env_name}", fontsize=36)
        ax.set_xlabel("Episode", fontsize=36)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax.grid(True, linestyle="--", alpha=0.7)

        # 构造图例句柄
        raw_handles = [
            plt.Line2D([0], [0], color=color, lw=2, alpha=0.3)
            for agent_name, color in agent_color_map.items()
        ]
        avg_handles = [
            plt.Line2D([0], [0], color=color, lw=2)
            for agent_name, color in agent_color_map.items()
        ]

        legend_labels = list(agent_color_map.keys())
        label_types = [" (Raw)", " (Smoothed)"]
        legend_handles = []
        for raw_handle in raw_handles:
            legend_handles.append(raw_handle)
        for avg_handle in avg_handles:
            legend_handles.append(avg_handle)
        font_properties = {"family": "Consolas", "size": 16}
        # 添加图例到每个子图的右下角
        ax.legend(
            legend_handles,
            [
                label + label_type
                for label_type in label_types
                for label in legend_labels
            ],
            loc="lower right",
            fontsize=16,
            ncol=2,
            prop=font_properties,
        )

    axs[0].set_ylabel("Training Rewards", fontsize=36)

    plt.tight_layout()
    plt.ylim(ymin, ymax)  # 设置y轴范围，可以根据实际数据调整
    plt.savefig(f"assets/{file_name}")
    plt.close()

def save_test_scores(all_test_scores, filename='assets/test_scores.csv'):
    df = pd.DataFrame()
    for env_name, agents in all_test_scores.items():
        row = {'Environment': env_name}
        for agent_name, scores in agents.items():
            avg_score = np.mean(scores)
            row[agent_name] = avg_score
        
       
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

 
    df.set_index('Environment', inplace=True)
    df.to_csv(filename)
    print(f"Test scores saved to {filename}")

def save_test_fill_rates(all_test_demands, all_test_satisfied_demands, filename='assets/test_fill_rates.csv'):
    
    df = pd.DataFrame()
    for env_name, agents in all_test_demands.items():
        row = {'Environment': env_name}
        for agent_name, results in agents.items():
            demand = results
            satisfied_demand = all_test_satisfied_demands[env_name][agent_name]
            fill_rates = []
            
            for episode_demand, episode_satisfied_demand in zip(demand, satisfied_demand):
                fill_rate = np.mean((np.array(episode_satisfied_demand) + 1e-5) / (np.array(episode_demand) + 1e-5))
                fill_rates.append(fill_rate)
            avg_fill_rate = np.mean(fill_rates)
            row[agent_name] = avg_fill_rate
        
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.set_index('Environment', inplace=True)
    df.to_csv(filename)
    print(f"Fill rates saved to {filename}")

def plot_test_inventories(all_test_inventories):
    
    sns.set_theme(style="whitegrid")
    sns.set_theme(font="serif")
    plt.rcParams["font.family"] = "serif"

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(33, 8))


    # 获取所有Agent的名称
    all_agent_names = set()
    for agents_inventories in all_test_inventories.values():
        all_agent_names.update(agents_inventories.keys())
    all_agent_names = sorted(all_agent_names)

    # 为每个Agent分配不同的颜色
    colors = sns.color_palette("husl", len(all_agent_names))
    agent_color_map = {
        agent_name: color for agent_name, color in zip(all_agent_names, colors)
    }

    for env_idx, (env_name, agents_inventories) in enumerate(all_test_inventories.items()):
        ax = axs[env_idx]

        for agent_name, inventories in agents_inventories.items():
            inventories = np.array(inventories)
            inventories = inventories.mean(axis=0)
            color = agent_color_map[agent_name]
            timestep = np.arange(len(inventories))
            sns.lineplot(x=timestep, y=inventories, ax=ax, color=color, linewidth=2)

        ax.set_title(f"{env_name}", fontsize=36)
        ax.set_xlabel("Timestep", fontsize=36)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax.grid(True, linestyle="--", alpha=0.7)

        
        raw_handles = [
            plt.Line2D([0], [0], color=color, lw=2)
            for agent_name, color in agent_color_map.items()
        ]

        legend_labels = list(agent_color_map.keys())
        legend_handles = []
        for raw_handle in raw_handles:
            legend_handles.append(raw_handle)
        font_properties = {"family": "Consolas", "size": 16}
        # 添加图例到每个子图的右下角
        ax.legend(
            legend_handles,
            [label for label in legend_labels],
            loc="lower right",
            fontsize=16,
            ncol=1,
            prop=font_properties,
        )

    axs[0].set_ylabel("Avg. Inventory", fontsize=36)

    plt.tight_layout()
    plt.savefig("assets/multi_env_multi_agent_test_inventories.png")
    plt.close()



def plot_test_orders(all_test_orders):
    
    sns.set_theme(style="whitegrid")
    sns.set_theme(font="serif")
    plt.rcParams["font.family"] = "serif"

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(33, 8))


    # 获取所有Agent的名称
    all_agent_names = set()
    for agents_orders in all_test_orders.values():
        all_agent_names.update(agents_orders.keys())
    all_agent_names = sorted(all_agent_names)

    # 为每个Agent分配不同的颜色
    colors = sns.color_palette("husl", len(all_agent_names))
    agent_color_map = {
        agent_name: color for agent_name, color in zip(all_agent_names, colors)
    }

    for env_idx, (env_name, agents_orders) in enumerate(all_test_orders.items()):
        ax = axs[env_idx]

        for agent_name, orders in agents_orders.items():
            orders = np.array(orders)
            orders = orders.mean(axis=0)
            color = agent_color_map[agent_name]
            timesteps = np.arange(len(orders))
            sns.lineplot(x=timesteps, y=orders, ax=ax, color=color, linewidth=2)

        ax.set_title(f"{env_name}", fontsize=36)
        ax.set_xlabel("Timestep", fontsize=36)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax.grid(True, linestyle="--", alpha=0.7)

        
        raw_handles = [
            plt.Line2D([0], [0], color=color, lw=2)
            for agent_name, color in agent_color_map.items()
        ]

        legend_labels = list(agent_color_map.keys())
        legend_handles = []
        for raw_handle in raw_handles:
            legend_handles.append(raw_handle)
        font_properties = {"family": "Consolas", "size": 16}
        # 添加图例到每个子图的右下角
        ax.legend(
            legend_handles,
            [label for label in legend_labels],
            loc="lower right",
            fontsize=16,
            ncol=1,
            prop=font_properties,
        )

    axs[0].set_ylabel("Avg. Order", fontsize=36)

    plt.tight_layout()
    plt.savefig("assets/multi_env_multi_agent_test_orders.png")
    plt.close()

def plot_test_satisfied_demands(all_test_satisfied_demands, expected_demands):
    sns.set_theme(style="whitegrid")
    sns.set_theme(font="serif")
    plt.rcParams["font.family"] = "serif"

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(33, 8))


    # 获取所有Agent的名称
    all_agent_names = set()
    for agents_demands in all_test_satisfied_demands.values():
        all_agent_names.update(agents_demands.keys())
    all_agent_names = sorted(all_agent_names)

    # 为每个Agent分配不同的颜色
    colors = sns.color_palette("husl", len(all_agent_names))
    agent_color_map = {
        agent_name: color for agent_name, color in zip(all_agent_names, colors)
    }

    for env_idx, (env_name, agents_demands) in enumerate(all_test_satisfied_demands.items()):
        ax = axs[env_idx]

        for agent_name, demands in agents_demands.items():
            demands = np.array(demands)
            demands = demands.mean(axis=0)
            color = agent_color_map[agent_name]
            timesteps = np.arange(len(demands))
            sns.lineplot(x=timesteps, y=demands, ax=ax, color=color, linewidth=2)
        env_expected_demands = expected_demands[env_name]
        timesteps = np.arange(len(env_expected_demands))
        sns.lineplot(x=timesteps, y=env_expected_demands, ax=ax, color="black", linewidth=2)


        ax.set_title(f"{env_name}", fontsize=36)
        ax.set_xlabel("Timestep", fontsize=36)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax.grid(True, linestyle="--", alpha=0.7)

        
        raw_handles = [
            plt.Line2D([0], [0], color=color, lw=2)
            for agent_name, color in agent_color_map.items()
        ]
        raw_handles.append(plt.Line2D([0], [0], color="black", lw=2))

        legend_labels = list(agent_color_map.keys())
        legend_labels.append("Expected Demand")
        legend_handles = []
        for raw_handle in raw_handles:
            legend_handles.append(raw_handle)
        font_properties = {"family": "Consolas", "size": 16}
        # 添加图例到每个子图的右下角
        ax.legend(
            legend_handles,
            [label for label in legend_labels],
            loc="lower right",
            fontsize=16,
            ncol=1,
            prop=font_properties,
        )

    axs[0].set_ylabel("Avg. Satisfied Demand", fontsize=36)

    

    plt.tight_layout()
    plt.savefig("assets/multi_env_multi_agent_test_satisfied_demands.png")
    plt.close()

def plot_test_fill_rates(all_test_satisfied_demands, all_test_demands):
    sns.set_theme(style="whitegrid")
    sns.set_theme(font="serif")
    plt.rcParams["font.family"] = "serif"

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(33, 8))


    # 获取所有Agent的名称
    all_agent_names = set()
    for agents_demands in all_test_satisfied_demands.values():
        all_agent_names.update(agents_demands.keys())
    all_agent_names = sorted(all_agent_names)

    # 为每个Agent分配不同的颜色
    colors = sns.color_palette("husl", len(all_agent_names))
    agent_color_map = {
        agent_name: color for agent_name, color in zip(all_agent_names, colors)
    }

   

    for env_idx, (env_name, agents_demands) in enumerate(all_test_demands.items()):
        ax = axs[env_idx]

        for agent_name, demands in agents_demands.items():
            demands = np.array(demands)
            demands = demands.mean(axis=0)
            color = agent_color_map[agent_name]
            timesteps = np.arange(len(demands))
            satisfied_demands = np.array(all_test_satisfied_demands[env_name][agent_name])
            satisfied_demands = satisfied_demands.mean(axis=0)

            fill_rates = (satisfied_demands + 1e-5) / (demands + 1e-5)
            sns.lineplot(x=timesteps, y=fill_rates, ax=ax, color=color, linewidth=2)
        

        ax.set_title(f"{env_name}", fontsize=36)
        ax.set_xlabel("Timestep", fontsize=36)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax.grid(True, linestyle="--", alpha=0.7)

        
        raw_handles = [
            plt.Line2D([0], [0], color=color, lw=2)
            for agent_name, color in agent_color_map.items()
        ]
        raw_handles.append(plt.Line2D([0], [0], color="black", lw=2))

        legend_labels = list(agent_color_map.keys())
        legend_handles = []
        for raw_handle in raw_handles:
            legend_handles.append(raw_handle)
        font_properties = {"family": "Consolas", "size": 16}
        # 添加图例到每个子图的右下角
        ax.legend(
            legend_handles,
            [label for label in legend_labels],
            loc="lower right",
            fontsize=16,
            ncol=1,
            prop=font_properties,
        )

    axs[0].set_ylabel("Fill Rate", fontsize=36)

    

    plt.tight_layout()
    plt.savefig("assets/multi_env_multi_agent_test_fill_rates.png")
    plt.close()

def plot_different_envs(env_simple: EnvSimple, env_seasonal: EnvSeasonal, env_complex: EnvComplex):

    seasonal_demands = env_seasonal.get_demand()
    fixed_demands = env_simple.get_demand()
    sns.set_theme(style="whitegrid")
    sns.set_theme(font="serif")
    plt.rcParams["font.family"] = "serif"

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(33, 8))


    # 为每个Agent分配不同的颜色
    colors = sns.color_palette("husl", 2)
    ax0 = axs[0]
    timesteps = np.arange(len(fixed_demands))
    sns.lineplot(x=timesteps, y=fixed_demands, ax=ax0, color=colors[0], linewidth=2)
    sns.lineplot(x=timesteps, y=seasonal_demands, ax=ax0, color=colors[1], linewidth=2)

    ax0.set_title("Expected Demand", fontsize=36)
    ax0.set_xlabel("Timestep", fontsize=36)
    ax0.tick_params(axis="both", which="major", labelsize=24)
    ax0.grid(True, linestyle="--", alpha=0.7)

        
    raw_handles = [
        plt.Line2D([0], [0], color=colors[0], lw=2),
        plt.Line2D([0], [0], color=colors[1], lw=2)
    ]
       
    legend_labels = ["EnvSimple", "EnvSeasonal\nEnvComplex"]
    legend_handles = []
    for raw_handle in raw_handles:
        legend_handles.append(raw_handle)
    font_properties = {"family": "Consolas", "size": 16}
    ax0.legend(
        legend_handles,
        [label for label in legend_labels],
        loc="upper right",
        fontsize=16,
        ncol=1,
        prop=font_properties,
    )

    fixed_hs, fixed_cs = env_simple.get_costs()
    seasonal_hs, seasonal_cs = env_complex.get_costs()

    colors = sns.color_palette("husl", 2)
    ax1 = axs[1]
    timesteps = np.arange(len(fixed_hs))
    sns.lineplot(x=timesteps, y=fixed_hs, ax=ax1, color=colors[0], linewidth=2)
    sns.lineplot(x=timesteps, y=seasonal_hs, ax=ax1, color=colors[1], linewidth=2)
    
    ax1.set_title("Holding Cost", fontsize=36)
    ax1.set_xlabel("Timestep", fontsize=36)
    ax1.tick_params(axis="both", which="major", labelsize=24)
    ax1.grid(True, linestyle="--", alpha=0.7)

    legend_handles = []
    raw_handles = [
        plt.Line2D([0], [0], color=colors[0], lw=2),
        plt.Line2D([0], [0], color=colors[1], lw=2),
    ]
    legend_labels = ["EnvSimple\nEnvSeasonal","EnvComplex"]
    for raw_handle in raw_handles:
        legend_handles.append(raw_handle)
    font_properties = {"family": "Consolas", "size": 16}
    ax1.legend(
        legend_handles,
        [label for label in legend_labels],
        loc="upper right",
        fontsize=16,
        ncol=1,
        prop=font_properties,
    )


    ax2 = axs[2]
    timesteps = np.arange(len(fixed_cs))
    sns.lineplot(x=timesteps, y=fixed_cs, ax=ax2, color=colors[0], linewidth=2)
    sns.lineplot(x=timesteps, y=seasonal_cs, ax=ax2, color=colors[1], linewidth=2)
    
    ax2.set_title("Lost Sales Cost", fontsize=36)
    ax2.set_xlabel("Timestep", fontsize=36)
    ax2.tick_params(axis="both", which="major", labelsize=24)
    ax2.grid(True, linestyle="--", alpha=0.7)

    legend_handles = []
    raw_handles = [
        plt.Line2D([0], [0], color=colors[0], lw=2),
        plt.Line2D([0], [0], color=colors[1], lw=2),
    ]
    legend_labels = ["EnvSimple\nEnvSeasonal","EnvComplex"]
    for raw_handle in raw_handles:
        legend_handles.append(raw_handle)
    font_properties = {"family": "Consolas", "size": 16}
    ax2.legend(
        legend_handles,
        [label for label in legend_labels],
        loc="upper right",
        fontsize=16,
        ncol=1,
        prop=font_properties,
    )

    
    plt.tight_layout()
    plt.savefig("assets/envs.png")
    plt.close()

import numpy as np

def generate_mock_demands(num_agents=3, num_timesteps=50, num_envs=3):
    """
    生成模拟的需求数据，结构为：
    {
        'Env1': {
            'Agent1': demand_array,
            'Agent2': demand_array,
            ...
        },
        'Env2': {...},
        ...
    }
    每个 demand_array 是一个二维数组，形状为 (num_episodes, num_timesteps)
    这里模拟为正态分布随机数，代表多个episode的需求数据。
    """
    env_names = [f"Env{i+1}" for i in range(num_envs)]
    agent_names = [f"Agent{i+1}" for i in range(num_agents)]
    num_episodes = 20  # 模拟的episode数量

    all_test_satisfied_demands = {}
    for env in env_names:
        agents_demands = {}
        for agent in agent_names:
            # 模拟需求数据，均值和方差可根据环境和agent调整
            mean_demand = np.random.uniform(20, 50)
            std_demand = np.random.uniform(5, 15)
            demands = np.random.normal(loc=mean_demand, scale=std_demand, size=(num_episodes, num_timesteps))
            demands = np.clip(demands, 0, None)  # 需求不能为负
            agents_demands[agent] = demands
        all_test_satisfied_demands[env] = agents_demands

    # 生成预期需求，结构为 {env_name: np.array}
    expected_demands = {}
    for env in env_names:
        expected_demands[env] = np.linspace(30, 40, num_timesteps)

    return all_test_satisfied_demands, expected_demands

