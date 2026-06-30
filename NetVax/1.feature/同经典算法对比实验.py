# %% 引入包
import scipy.io as scio
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

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



# %%　KNN分类器
def KNN_classifier(features, label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    accuracy = cross_val_score(knn, features, label.ravel(), cv=skf, scoring='accuracy')
    return accuracy


# %% XGBoost特征选择
def XGBoost_FS(dataFile):
    # 读取数据
    if dataFile.endswith('.arff'):
        features, label = read_file_arff(dataFile)
    else:
        features, label = read_file_mat(dataFile)

    # 特征选择
    start = time.time()
    model = XGBClassifier()
    model.fit(features, label)
    importance = model.feature_importances_
    thre
    index = np.argsort(-importance)[:num_features]
    features = features[:, index]
    end = time.time()
    return features, label, end-start


# 随机森林特征选择
def RandomForest_FS(dataFile):
    # 读取数据
    if dataFile.endswith('.arff'):
        features, label = read_file_arff(dataFile)
    else:
        features, label = read_file_mat(dataFile)

    # 特征选择
    start = time.time()
    model = RandomForestClassifier()
    model.fit(features, label)
    importance = model.feature_importances_
    index = np.argsort(-importance)[:num_features]
    features = features[:, index]
    end = time.time()
    return features, label, end-start

# %%

dataFile = 'data\\KeChen\\Leukemia_1.mat'


features, label, tim = XGBoost_FS(dataFile, num_features=)
accuracy = KNN_classifier(features, label)


# %%
