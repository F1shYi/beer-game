models/里面保存的是训练好的模型参数和模型的训练日志
assets_*/里面保存的是测试的数据，分别在holding cost为高和低（原始和乘以0.1）的情况下。

想要得到测试数据，直接跑python eval.py即可，可以调节plot_train_curves()和test()的参数来选择是对高holding cost的情况测试还是低holding cost的情况测试

想要训练，跑python main.py --env [env_name] --agent [agent_name]，具体的选项见train_parser()里的定义。


我设计的不同环境的定义在environment.py里封装好，分别是EnvSimple, EnvSeasonal, EnvComplex。holding cost和lost sales cost在前两个环境为定值，最后一个环境为time-varying。Expected consumer demand在第一个环境为定值，后两个环境为time-varying。以模拟真实供应链情景。

QNet, Actor, Critic的网络架构在model.py里定义，均为简单的MLP

agent.py里实现了DQN, PPO, SAC三种算法。训练的代码参考了https://hrl.boyuai.com/

main.py封装了训练流程。

eval.py是绘制训练曲线，测试曲线，测试数据等。

具体需要用到的函数实现在utils.py里