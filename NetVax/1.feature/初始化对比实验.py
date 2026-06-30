# %%
import numpy as np
import pandas as pd

# %%  初始化种群
population_size = 100
problem_size = 2

pop_random = np.random.uniform(low = 0, high=1, size=(population_size, problem_size))


# %% Tent映射
def tent_map(x, r):
    if x < r:
        return x / r
    else:
        return (1 - x) / (1 - r)

def initialize_population(size, r, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    population = np.empty((size, 2))
    population[0] = np.random.rand(2)  # 初始种群随机化
    
    for i in range(1, size):
        x_new = tent_map(population[i-1][0], r)
        y_new = tent_map(population[i-1][1], r)
        population[i] = [x_new, y_new]
    
    return population

# 用法示例
population_size = 100
r_parameter = 0.8
seed_value = 42

pop_tent = initialize_population(population_size, r_parameter, seed_value)


# %% 佳点集初始化种群
# 读取matlab.mat文件
import scipy.io as sio
mat_contents = sio.loadmat('matlab.mat')
good_point_set = mat_contents['pop2']




# %% 根据good_point_set和pop_tent绘制散点图

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(good_point_set[:, 0], good_point_set[:, 1], s = 50,
            c = 'r', marker='s')
plt.scatter(pop_tent[:, 0], pop_tent[:, 1], s = 70,
            c = '#FFBBFF', marker='^')
plt.rcParams.update({'font.size': 17})
plt.rcParams['font.family']=['SimHei']#
plt.legend(labels=['Good point set initialization','Tent Map Initialization'])
plt.show()




# %% 根据good_point_set和pop_random绘制散点图
# 绘制good_point_set的散点图
import matplotlib.pyplot as plt
markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

plt.figure(figsize=(8, 6))


plt.scatter(good_point_set[:, 0], good_point_set[:, 1], s = 50,
            c = 'r', marker='s', label='Good Point Population')
plt.scatter(pop_random[:, 0], pop_random[:, 1], s = 70,
            c = '#CD5C5C', marker='o', label='Random Population')
plt.rcParams.update({'font.size': 17})
plt.rcParams['font.family']=['SimHei']#
plt.legend(labels=['佳点集初始化','随机初始化'])
plt.show()





# %% 第二张图：柯西分布与高斯分布图
import numpy as np
import matplotlib.pyplot as plt
# 使用宋体字体

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


# 设置x值的范围
x = np.linspace(-10, 10, 1000)

# 计算柯西分布和高斯分布的概率密度函数
cauchy_pdf = 1 / (np.pi * (1 + x**2))
gaussian_pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)

# 绘制曲线
plt.figure(figsize=(8, 6))

# 字体变大
plt.rcParams.update({'font.size': 15})
plt.plot(x, cauchy_pdf, label=u'柯西分布概率密度函数', )
plt.plot(x, gaussian_pdf, label=u'高斯分布概率密度函数')
plt.xlabel('x')
plt.ylabel('概率密度', fontsize=20)
plt.title('柯西分布与高斯分布', fontsize=20)
plt.legend()
plt.show()


# %% 第二张图【英文版】：柯西分布与高斯分布图
import numpy as np
import matplotlib.pyplot as plt



# 设置x值的范围
x = np.linspace(-10, 10, 1000)

# 计算柯西分布和高斯分布的概率密度函数
cauchy_pdf = 1 / (np.pi * (1 + x**2))
gaussian_pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)

# 绘制曲线
plt.figure(figsize=(8, 6))

# 字体变大
plt.rcParams.update({'font.size': 15})
plt.plot(x, cauchy_pdf, color='green', linewidth=2, label=u'Cauchy', )
plt.plot(x, gaussian_pdf, color='red', linewidth=2, label=u'Gaussian')
plt.xlabel('X')
plt.ylabel('Probability dXensity', fontsize=20)
plt.title('Probability density function', fontsize=20)
plt.legend()
plt.show()

# %%
