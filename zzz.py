# import torch
# import numpy as np

# a = torch.tensor([[1], [2], [3]])
# b = torch.tensor([[5], [6], [7]])
# c = torch.tensor([[9], [4], [0]])

# f = a.mul_(b)


# d = torch.stack([a, b, c])
# e = torch.cat([a, b, c])

# l = [a.numpy(), b.numpy()]

# l.append(c.numpy())

# l = np.array(l).flatten()

# x = torch.cat(l)
# y = torch.stack(l)

# print(l)

# print(d.size())
# print(e.size())



import torch


a = torch.tensor([2e-13])

b = a

print(b)








# import numpy as np # 快速操作结构数组的工具
# import matplotlib.pyplot as plt  # 可视化绘制
# from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取


# # 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型
# data=[
#     [0.067732,3.176513],[0.427810,3.816464],[0.995731,4.550095],[0.738336,4.256571],[0.981083,4.560815],
#     [0.526171,3.929515],[0.378887,3.526170],[0.033859,3.156393],[0.132791,3.110301],[0.138306,3.149813],
#     [0.247809,3.476346],[0.648270,4.119688],[0.731209,4.282233],[0.236833,3.486582],[0.969788,4.655492],
#     [0.607492,3.965162],[0.358622,3.514900],[0.147846,3.125947],[0.637820,4.094115],[0.230372,3.476039],
#     [0.070237,3.210610],[0.067154,3.190612],[0.925577,4.631504],[0.717733,4.295890],[0.015371,3.085028],
#     [0.335070,3.448080],[0.040486,3.167440],[0.212575,3.364266],[0.617218,3.993482],[0.541196,3.891471]
# ]

# #生成X和y矩阵
# dataMat = np.array(data)
# X = dataMat[:,0:1] # 2D
# y = dataMat[:,1]

# # ========Lasso回归========
# model = Lasso(alpha=0.001, tol=0.000000000000000000001)  # 调节alpha可以实现对拟合的程度
# # model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
# # model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
# model.fit(X, y)   # 线性回归建模
# print('系数矩阵:\n',model.coef_)
# print('线性回归模型:\n',model)
# # print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效
# # 使用模型预测
# predicted = model.predict(X)

# # 绘制散点图 参数：x横轴 y纵轴
# plt.scatter(X, y, marker='x')
# plt.plot(X, predicted,c='r')

# # 绘制x轴和y轴坐标
# plt.xlabel("x")
# plt.ylabel("y")

# # 显示图形
# plt.savefig('Lasso.png')







# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.linear_model import MultiTaskLasso, Lasso

# rng = np.random.RandomState(42)
# # ===========================产生模拟样本数据=========================
# # 用随机的频率、相位产生正弦波的二维系数
# n_samples, n_features, n_tasks = 100, 30, 40  # n_samples样本个数，n_features特征个数，n_tasks估计值的个数
# n_relevant_features = 5 # 自定义实际有用特征的个数
# coef = np.zeros((n_tasks, n_features)) # 系数矩阵的维度

# times = np.linspace(0, 2 * np.pi, n_tasks)
# for k in range(n_relevant_features):
#     coef[:, k] = np.sin((1. + rng.randn(1)) * times + 3 * rng.randn(1)) # 自定义数据矩阵，用来生成模拟输出值

# X = rng.randn(n_samples, n_features)  # 产生随机输入矩阵
# Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks) # 输入*系数+噪声=模拟输出
# # ==============================使用样本数据训练系数矩阵============================
# coef_lasso_ = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])
# coef_multi_task_lasso_ = MultiTaskLasso(alpha=1.).fit(X, Y).coef_  # 多任务训练

# # #############################################################################
# # Plot support and time series
# fig = plt.figure(figsize=(8, 5))
# plt.subplot(1, 2, 1)
# plt.spy(coef_lasso_)
# plt.xlabel('Feature')
# plt.ylabel('Time (or Task)')
# plt.text(10, 5, 'Lasso')
# plt.subplot(1, 2, 2)
# plt.spy(coef_multi_task_lasso_)
# plt.xlabel('Feature')
# plt.ylabel('Time (or Task)')
# plt.text(10, 5, 'MultiTaskLasso')
# fig.suptitle('Coefficient non-zero location')

# feature_to_plot = 0
# plt.figure()
# lw = 2
# plt.plot(coef[:, feature_to_plot], color='seagreen', linewidth=lw,
#          label='Ground truth')
# plt.plot(coef_lasso_[:, feature_to_plot], color='cornflowerblue', linewidth=lw,
#          label='Lasso')
# plt.plot(coef_multi_task_lasso_[:, feature_to_plot], color='gold', linewidth=lw,
#          label='MultiTaskLasso')
# plt.legend(loc='upper center')
# plt.axis('tight')
# plt.ylim([-1.1, 1.1])
# plt.savefig('MultiTaskLasso.png')

# import torch
# import numpy as np
# import random

# class Round(torch.autograd.Function):
#     @staticmethod
#     def forward(self, input):
#         output = torch.round(input)
#         return output
#     @staticmethod
#     def backward(self, grad_input):
#         grad_output = grad_input
#         return grad_output

# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)

# x = torch.randn(2, 3, requires_grad=True)
# # y = torch.round(x)
# y = Round.apply(x)
# y = 2 * y

# y = y.sum()
# y.backward()

# print(x.grad)
