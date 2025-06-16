import numpy as np


class EnvBase:
    def __init__(
        self, num_firms, p, h, c, initial_inventory, poisson_lambda=10, max_steps=100
    ):
        """
        初始化供应链管理仿真环境。

        :param num_firms: 企业数量
        :param p: 各企业的价格列表
        :param h: 库存持有成本
        :param c: 损失销售成本
        :param initial_inventory: 每个企业的初始库存
        :param poisson_lambda: 最下游企业需求的泊松分布均值
        :param max_steps: 每个episode的最大步数
        """
        self.num_firms = num_firms
        self.p = p  # 企业的价格列表
        self.h = h  # 库存持有成本
        self.c = c  # 损失销售成本
        self.poisson_lambda = poisson_lambda  # 泊松分布的均值

        self.max_steps = max_steps  # 每个episode的最大步数
        self.initial_inventory = initial_inventory  # 初始库存

        # 初始化库存
        self.inventory = np.full((num_firms, 1), initial_inventory)
        # 初始化订单量
        self.orders = np.zeros((num_firms, 1))
        # 初始化已满足的需求量
        self.satisfied_demand = np.zeros((num_firms, 1))
        # 记录当前步数
        self.current_step = 0
        # 标记episode是否结束
        self.done = False

    def reset(self):
        """
        重置环境状态。
        """
        self.inventory = np.full((self.num_firms, 1), self.initial_inventory)
        self.orders = np.zeros((self.num_firms, 1))
        self.satisfied_demand = np.zeros((self.num_firms, 1))
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """
        获取每个企业的观察信息，包括订单量、满足的需求量和库存。
        每个企业的状态是独立的，包括自己观察的订单、需求和库存。
        """
        return np.concatenate(
            (self.orders, self.satisfied_demand, self.inventory), axis=1
        )

    def _update_demand(self):
        """
        根据规则生成每个企业的需求。
        最下游企业的需求遵循泊松分布，其他企业的需求等于下游企业的订单量。
        """
        raise NotImplementedError("")

    def _update_reward(self):
        raise NotImplementedError("")

    def get_name(self):
        raise NotImplementedError("")
    
    def get_demand(self):
        raise NotImplementedError("")
    
    def get_costs(self):
        raise NotImplementedError("")

    def step(self, actions):
        """
        执行一个时间步的仿真，根据给定的行动 (每个企业的订单量) 更新环境状态。

        :param actions: 每个企业的订单量 (shape: (num_firms, 1))，即每个智能体的行动
        :return: next_state, reward, done
        """
        self.orders = actions  # 更新订单量
        # 生成各企业的需求
        self.demand = self._update_demand()

        # 计算每个企业收到的订单量和满足的需求
        for i in range(self.num_firms):
            if i == 0:
                # 第一企业从外部需求直接得到满足
                self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])
            else:
                # 后续企业的需求由上游企业订单决定
                self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])

        # 更新库存
        for i in range(self.num_firms):
            self.inventory[i] = (
                self.inventory[i] + self.orders[i] - self.satisfied_demand[i]
            )

        # 计算每个企业的奖励: p_i * d_{it} - p_{i+1} * q_{it} - h * I_{it}
        rewards = self._update_reward()
        # 增加步数
        self.current_step += 1

        # 判断是否结束（比如达到最大步数）
        if self.current_step >= self.max_steps:
            self.done = True

        return self._get_observation(), rewards, self.done


class EnvSimple(EnvBase):
    def __init__(
        self, num_firms, p, h, c, initial_inventory, poisson_lambda=10, max_steps=100
    ):
        super().__init__(
            num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps
        )

    def _update_demand(self):
        demand = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            if i == 0:
                # 最下游企业的需求遵循泊松分布，均值为 poisson_lambda
                demand[i] = np.random.poisson(self.poisson_lambda)
            else:
                # 上游企业的需求等于下游企业的订单量
                demand[i] = self.orders[i - 1]  # d_{i+1,t} = q_{it}
        return demand
    
    def get_demand(self):
        return [self.poisson_lambda for _ in range(self.max_steps)]

    def get_costs(self):
        return [self.h for _ in range(self.max_steps)], [self.c for _ in range(self.max_steps)]

    def _update_reward(self):
        # 计算每个企业的奖励: p_i * d_{it} - p_{i+1} * q_{it} - h * I_{it}
        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))  # 损失销售费用

        for i in range(self.num_firms):
            rewards[i] += (
                self.p[i] * self.satisfied_demand[i]
                - (self.p[i + 1] if i + 1 < self.num_firms else 0) * self.orders[i]
                - self.h * self.inventory[i]
            )

            # 损失销售计算
            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c

        rewards -= loss_sales  # 总奖励扣除损失销售成本
        return rewards


class EnvSeasonal(EnvBase):
    def __init__(
        self,
        num_firms,
        p,
        h,
        c,
        initial_inventory,
        poisson_lambda=10,
        max_steps=100,
        seasonality_factor=0.5,
        impulse_magnitude=15,
        impulse_duration=10,
        impulse_start=70,
    ):
        super().__init__(
            num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps
        )
        self.seasonality_factor = seasonality_factor
        self.seasonal_wave = np.sin(np.linspace(0, 2 * np.pi, self.max_steps))

        # Impulse parameters
        self.impulse_magnitude = impulse_magnitude
        self.impulse_duration = impulse_duration
        self.impulse_start = impulse_start
        self.impulse_end = impulse_start + impulse_duration

    def _update_demand(self):
        demand = np.zeros((self.num_firms, 1))
        seasonal_lambda = self.poisson_lambda * (
            1 + self.seasonality_factor * self.seasonal_wave[self.current_step]
        )

        # Apply impulse effect
        if self.impulse_start <= self.current_step < self.impulse_end:
            impulse_effect = self.impulse_magnitude * (
                1 - (self.current_step - self.impulse_start) / self.impulse_duration
            )
            seasonal_lambda += impulse_effect

        for i in range(self.num_firms):
            if i == 0:
                # 最下游企业的需求遵循季节性调整和impulse效应的泊松分布
                demand[i] = np.random.poisson(seasonal_lambda)
            else:
                # 上游企业的需求等于下游企业的订单量
                demand[i] = self.orders[i - 1]
        return demand

    def get_demand(self):
        demands = [self.poisson_lambda * (1 + self.seasonality_factor * self.seasonal_wave[current_step]) for current_step in range(self.max_steps)]
        for step, _ in enumerate(demands):
            if self.impulse_start <= step <= self.impulse_end:
                impulse_effect = self.impulse_magnitude * (
                    1 - (step - self.impulse_start) / self.impulse_duration
                )
                demands[step] += impulse_effect
        return demands
    
    def get_costs(self):
        return super().get_costs()
    
    def _update_reward(self):
        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))

        for i in range(self.num_firms):
            rewards[i] += (
                self.p[i] * self.satisfied_demand[i]
                - (self.p[i + 1] if i + 1 < self.num_firms else 0) * self.orders[i]
                - self.h * self.inventory[i]
            )

            # 损失销售计算
            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c

        rewards -= loss_sales
        return rewards


class EnvComplex(EnvSeasonal):
    def __init__(
        self,
        num_firms,
        p,
        h,
        c,
        initial_inventory,
        poisson_lambda=10,
        max_steps=100,
        seasonality_factor=0.5,
        impulse_magnitude=15,
        impulse_duration=10,
        impulse_start=70,
        c_variation=0.5,
    ):
        super().__init__(
            num_firms,
            p,
            h,
            c,
            initial_inventory,
            poisson_lambda,
            max_steps,
            seasonality_factor,
            impulse_magnitude,
            impulse_duration,
            impulse_start,
        )
        self.h_base = h
        self.c_base = c
        self.c_variation = c_variation
        self.h = h
        self.c = c

    def _update_costs(self):
        self.h = self.h_base * (
            1 + self.seasonality_factor * self.seasonal_wave[self.current_step]
        )

        # 在 impulse 阶段增加损失销售成本并进行线性衰减
        if self.impulse_start <= self.current_step < self.impulse_end:
            impulse_effect = self.c_variation * (
                1 - (self.current_step - self.impulse_start) / self.impulse_duration
            )
            self.c = self.c_base * (1 + impulse_effect)
        else:
            self.c = self.c_base
    
    def get_costs(self):
        hs = []
        cs = []
        for current_step in range(self.max_steps):
            h = self.h_base * (1 + self.seasonality_factor * self.seasonal_wave[current_step])
            hs.append(h)
            if self.impulse_start <= current_step <= self.impulse_end:
                c = self.c_base * (1 + self.c_variation * (1 - (current_step - self.impulse_start) / self.impulse_duration))
            else:
                c = self.c_base
            cs.append(c)
        return hs, cs

    def step(self, actions):
        self._update_costs()
        return super().step(actions)

    

    def _update_reward(self):
        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))

        for i in range(self.num_firms):
            rewards[i] += (
                self.p[i] * self.satisfied_demand[i]
                - (self.p[i + 1] if i + 1 < self.num_firms else 0) * self.orders[i]
                - self.h * self.inventory[i]
            )

            # 损失销售计算
            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c

        rewards -= loss_sales
        return rewards
