models/里面保存的是训练好的模型参数和模型的训练日志
assets_*/里面保存的是测试的数据，分别在holding cost为高和低（原始和乘以0.1）的情况下。

想要得到测试数据，直接跑python eval.py即可，可以调节plot_train_curves()和test()的参数来选择是对高holding cost的情况测试还是低holding cost的情况测试

想要训练，跑python main.py --env [env_name] --agent [agent_name]，具体的选项见train_parser()里的定义。
