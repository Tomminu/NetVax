# %%
import scipy.io as scio
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
import torch
from torchmetrics.functional import pairwise_euclidean_distance
import time

# %% 读取.mat类型文件
def read_file_mat(dataFile):
    data = scio.loadmat(dataFile)
    features = data['X']
    label = data['Y']
    return (features, label)


# %% 读取.arff文件的数据
from scipy.io import arff
def read_file_arff(dataFile):
    data, meta = arff.loadarff(dataFile)
    data = pd.DataFrame(data)

    label = data.iloc[:,-1]
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)
    features = data.iloc[:,:-1].values
    return (features, label)


# %% 随机均匀初始化
def random_antibody_fcn2(population_size, problem_size, threshold):
    # 随机初始化种群
    population = np.random.uniform(low = -1, high=1, size=(population_size, problem_size))
    # 遍历种群，根据阈值进行二进制化
    for i in range(0, population_size):
        for j in range(0, problem_size):
            if population[i][j] > threshold:
                population[i][j] = 1
            else:
                population[i][j] = 0
    return population



# %% 柯西分布初始化种群
def random_antibody_fcn(population_size, problem_size, threshold=-0.2):
    # 生成柯西分布的随机数作为种群
    population = np.random.standard_cauchy(size=(population_size, problem_size))
    # 遍历种群，根据阈值进行二进制化
    for i in range(0, population_size):
        for j in range(0, problem_size):
            if population[i][j] > threshold:
                population[i][j] = 1
            else:
                population[i][j] = 0
    return population



# %%　KNN分类器
def KNN_classifier(antibody, features, label):
    # 获得一维矩阵中值为1的索引
    index = np.where(antibody == 1)
    # 获得features中对应的列
    features_selected = features[:, index[0]]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    accuracy = cross_val_score(knn, features_selected, label.ravel(), cv=skf, scoring='accuracy')
    return accuracy


# %% 计算亲和度
def calculate_affinity_fcn(_antibodies, features, label, w=0.99):
    problem_size = features.shape[1]
    antibodies = np.zeros((_antibodies.shape[0], _antibodies.shape[1]+1))

    for ind in range(len(_antibodies)):
        ratio = np.sum(_antibodies[ind]) / problem_size
        accuracy = KNN_classifier(_antibodies[ind], features, label)

        # 计算亲和度 
        accuracy = np.mean(accuracy)
        affinity = w * (1-accuracy) + (1 - w) * ratio
        antibodies[ind] = np.append(_antibodies[ind], affinity)
    return antibodies


# %% 克隆算法
def clone_antibodies_fcn(antibodies):
    clones = []
    aff = antibodies[:, -1]
    # aff越小，argsort越靠前，计算aff中各值的排序，返回索引
    aff = (len(aff) - np.argsort(aff))

    for a, index in zip(antibodies, range(len(antibodies))):
        # clones += [deepcopy(a[:-1]) for _ in range(round(-math.log(a[-1])))]
        clones += [deepcopy(a[:-1]) for _ in range(aff[index])]
    return np.array(clones)



# %% 突变策略
def mutation_fcn(clones, mutation_exp, alpha=0.8):
    # mut_pop = np.random.standard_cauchy(size = clones.shape)
    mut_pop = np.random.uniform(low=-1, high=1, size=clones.shape)
    threshold = 0.5 - mutation_exp*alpha
    mut_pop[mut_pop > threshold] = 1
    mut_pop[mut_pop <= threshold] = 0

    # 将nut_pop的0变1，1变0
    mut_pop = 1 - mut_pop
    # 同clones相乘
    clones = clones * mut_pop
    return clones




# %% Fisher score 特征重要性计算
def fisher_score(features, label):
    # 特征数-一般是高维特征
    num_features = features.shape[1]
    # 样本数量
    num_samples = features.shape[0]
    # 类别标签
    labels = np.unique(label)
    # 类别间离散度矩阵
    S_B = np.zeros(num_features)
    # 类别内离散度矩阵
    S_W = np.zeros(num_features)
    # 总体均值
    mean = np.mean(features, axis=0)
    
    for i in labels:
        # 获得每一类的索引
        index = np.where(label == i)
        # 根据index获取features中对应行的数据
        features_i = features[index,:][0]
        # 样本数量
        num_samples_i = features_i.shape[0]
        # 计算不同特征内的均值
        mean_i = np.mean(features_i, axis=0)
        # 计算类间方差
        S_B = S_B + (mean_i - mean) * (mean_i - mean) * num_samples_i / num_samples

        # mean_i纵向扩展num_samples_i行
        mean_i_tile = np.tile(mean_i, (num_samples_i,1))
        # 计算类内方差
        S_W = S_W + np.sum((features_i - mean_i_tile) * (features_i - mean_i_tile), axis=0) / num_samples
        
    # 计算Fisher score
    Fisher_score = S_B / S_W
    return Fisher_score




# %% Fisher Score 特征选择
def Fisher_FS(dataFile, num_features=200):
    # 读取数据
    if dataFile.endswith('.arff'):
        features, label = read_file_arff(dataFile)
    else:
        features, label = read_file_mat(dataFile)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    accuracy = cross_val_score(knn, features, label.ravel(), cv=skf, scoring='accuracy')
    print("Full accuracy:{}" .format(np.mean(accuracy)))

    # 计算Fisher score
    Fisher_score = fisher_score(features, label)
    # 找到Fisher score最大的前num_features个特征
    index = np.argsort(-Fisher_score)[:num_features]
    # 选取features中对应的列
    features = features[:, index]
    return features, label



#%% mRMR 特征选择
from  mrmr import mrmr_classif
def mRMR_FS(dataFile, num_features=200):
    if dataFile.endswith('.arff'):
        features, label = read_file_arff(dataFile)
    else:
        features, label = read_file_mat(dataFile)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    accuracy = cross_val_score(knn, features, label.ravel(), cv=skf, scoring='accuracy')
    print("Full accuracy:{}" .format(np.mean(accuracy)))

    index = mrmr_classif(pd.DataFrame(features), pd.DataFrame(label), K=num_features)
    features = features[:, index]
    return features, label


# %% set parameters
def run(features, label, problem_size, population_init=40, clone_rate=0.7, omega=0.9, alpha=0.8):
    _iteration = 0
    stop_condition = 30
    # population_init = 50
    # population_min = 20 
    # selection_size = 10
    
    # clone_rate = 0.5
    clone_size  = int(clone_rate * 10)

    _antibodies = []
    # 初始化种群
    _antibodies= random_antibody_fcn(population_init, problem_size, threshold=-0.2)
    # 计算亲和度并排序
    _antibodies = calculate_affinity_fcn(_antibodies, features, label, omega)
    _antibodies = _antibodies[np.argsort(_antibodies[:, -1])]


    while _iteration < stop_condition:
        _iteration += 1
        print('iteration: ', _iteration)

        # 克隆并突变、计算亲和度
        clones = clone_antibodies_fcn(_antibodies[:clone_size])
        # 突变 
        clones = mutation_fcn(clones, _iteration/stop_condition, alpha)
        clones = calculate_affinity_fcn(clones, features, label, omega)


        # 种群多样性
        antibodies2= random_antibody_fcn(population_init, problem_size, _iteration/stop_condition)
        antibodies2 = calculate_affinity_fcn(antibodies2, features, label, omega)
        # 计算种群数量
        # pop_num = np.round(population_init-population_min)*(_iteration/stop_condition) + population_min
        pop_num = population_init
        
        # 合并种群、亲和度排序、选择前pop_num个
        _antibodies = np.append(_antibodies, clones, axis=0)
        _antibodies = np.append(_antibodies, antibodies2, axis=0)
        _antibodies = _antibodies[np.argsort(_antibodies[:, -1])]
        _antibodies = _antibodies[:int(pop_num)]

    # 选择最优的抗体
    best_antibody = _antibodies[0]
    # 计算最优抗体的准确率和特征数
    accuracy = KNN_classifier(best_antibody, features, label)
    feaSize = sum(best_antibody[:-1])
    # return accuracy, feaSize
    return (accuracy, feaSize, best_antibody)

  


# %% read data 
# dataFile = 'data\\Weka\\SRBCT.arff'
dataFile = 'data\\KeChen\\Leukemia_1.mat'
# features, label = mRMR_FS(dataFile, num_features=200)
features, label = Fisher_FS(dataFile, num_features=200)
problem_size = features.shape[1]

# %% 自适应突变率参数选择实验
alpha = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
acc, feaSize = [], []
for i in alpha:
    acctemp, fsizetemp = [], []
    for j  in range(5):
        accuracy, fsize = run(features, label, problem_size, population_init=40, 
                              clone_rate=0.7, omega=0.99, alpha=i)
        acctemp.append(np.mean(accuracy))
        fsizetemp.append(fsize)
    acc.append(np.mean(acctemp))
    feaSize.append(np.mean(fsizetemp))




#%% 亲和度函数参数
omega = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
acc, feaSize = [], []
for i in omega:
    acctemp, fsizetemp = [], []
    for j  in range(5):
        accuracy, fsize = run(features, label, problem_size, population_init=40, clone_rate=0.7, omega=i)
        acctemp.append(np.mean(accuracy))
        fsizetemp.append(fsize)
    acc.append(np.mean(acctemp))
    feaSize.append(np.mean(fsizetemp))

    
# %% 突变率参数选择实验跑五次
clone_rate = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
acc, feaSize, = [], []
for i in clone_rate:
    acctemp, fsizetemp = [], []
    for j  in range(5):
        accuracy, fsize = run(features, label, problem_size, population_init=40, clone_rate=i)
        acctemp.append(np.mean(accuracy))
        fsizetemp.append(fsize)
    acc.append(np.mean(acctemp))
    feaSize.append(np.mean(fsizetemp))



# %% 种群大小参数选择实验
population_init = [10,20,30,40,50,60,70]
acc, accVar, feaSize, feaVar = [], [], [], []
for i in population_init:
    accuracy, fsize = run(features, label, problem_size, population_init=i)
    acc.append(np.mean(accuracy))
    accVar.append(np.var(accuracy))
    feaSize.append(fsize)
    feaVar.append(np.var(fsize))
    print("acc:{}, acc-var:{}, feaSize:{}, feaS-var:{}".format
          (np.mean(accuracy), np.var(accuracy), fsize, np.var(fsize)))



# %% 稳定性实验
acc, accVar, feaSize, feaVar, antibodies = [], [], [], [],[]
for i in range(5):
    accuracy, fsize, antibody = run(features, label, problem_size)
    acc.append(np.mean(accuracy))
    accVar.append(np.var(accuracy))
    feaSize.append(fsize)
    feaVar.append(np.var(fsize))
    antibodies.append(antibody)

# 找到antibodies中为1的索引
new_anti = []
for anti in antibodies:
    index = np.where(anti == 1)
    new_anti.append(index[0])

















# %% 存储acc, feaSize, accVar, feaVar
import pandas as pd
data = {'acc':acc, 'accVar':accVar, 'feaSize':feaSize, 'feaVar':feaVar}
df = pd.DataFrame(data)
df.to_csv('DLBCL.csv', index=False)
# %% 存储acc, feaSize, accVar, feaVar, new_anti
import pandas as pd
data = {'acc':acc, 'accVar':accVar, 'feaSize':feaSize, 'feaVar':feaVar, 'antibodies':new_anti}
df = pd.DataFrame(data)
df.to_csv('Leukemia_1.csv', index=False)

