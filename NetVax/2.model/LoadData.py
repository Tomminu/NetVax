# %%
import os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import tree


# %% 数据预处理--剔除重复值(kdd有部分正常样本与异常样本重叠)
def delete_redundant_data(total_data):
    print("原始数据集大小：", total_data.shape)
    # 删除特征值重合的样本
    feature = total_data.iloc[:, :-1].drop_duplicates()
    total_data = total_data.loc[feature.index]
    print("去除重复数据后的数据集大小：", total_data.shape)
    return total_data


# %% 加载NSL_KDD数据集
def load_NSL_KDD():
    header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type',
                    'success_pred']
    train_df = pd.read_csv("NSL-KDD/KDDTrain+.txt", names=header_names, header=None)
    test_df = pd.read_csv("NSL-KDD/KDDTest+.txt", names=header_names, header=None)

    total_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)  # 行拼接
    total_data = delete_redundant_data(total_data)

    # 转为二分类
    total_data['label'] = [0 if total_data.attack_type[i]
                                == 'normal' else 1 for i in total_data.index]
    # le = LabelEncoder()
    # total_data['label'] = le.fit_transform(total_data['attack_type'])   

    # 特征选择
    imp_feature = ['flag', 'src_bytes', 'protocol_type', 'dst_host_same_srv_rate',
                   'dst_host_srv_count', 'num_failed_logins', 'dst_host_rerror_rate',
                   'dst_host_diff_srv_rate', 'count', 'logged_in', 'same_srv_rate',
                   'dst_host_srv_serror_rate', 'hot', 'serror_rate', 'service',
                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                   'srv_serror_rate', 'is_guest_login', 'diff_srv_rate', 'dst_host_count',
                   'dst_host_same_src_port_rate', 'dst_host_srv_rerror_rate', 'duration']
    feature_data = total_data[imp_feature]
    label_data = total_data['label'].values

    # one_hot编码
    feature_data = pd.get_dummies(data=feature_data, columns=['protocol_type', 'service', 'flag'])

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data = scaler.fit_transform(feature_data)

    return feature_data, label_data


# %% 加载NSL_KDD数据集特征全集
def load_NSL_KDD_allFeature():
    header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type',
                    'success_pred']
    train_df = pd.read_csv("NSL-KDD/KDDTrain+.txt", names=header_names, header=None)
    test_df = pd.read_csv("NSL-KDD/KDDTest+.txt", names=header_names, header=None)

    total_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)  # 行拼接
    total_data = delete_redundant_data(total_data)  # 去冗余

    # 转为二分类
    total_data['label'] = [0 if total_data.attack_type[i]
                                == 'normal' else 1 for i in total_data.index]

    total_data.drop(['attack_type'], axis=1, inplace=True)  # 去掉细分攻击类型那一列
    total_data.drop(['success_pred'], axis=1, inplace=True)  # 去掉最后一列score-the severity of the traffic input itself

    # 特征选择
    feature_data = total_data.iloc[:, :-1]
    label_data = total_data['label'].values

    # feature_data的one_hot编码
    feature_data = pd.get_dummies(data=feature_data)

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data = scaler.fit_transform(feature_data)

    return feature_data, label_data


# %%  加载UNSW_NB15数据集
def load_UNSW_NB15():
    traindata = pd.read_csv('UNSW_NB15/UNSW_NB15_training-set.csv', index_col=False)
    testdata = pd.read_csv('UNSW_NB15/UNSW_NB15_testing-set.csv', index_col=False)

    unsw = pd.concat([traindata, testdata], axis=0, ignore_index=True)  # traindata与testdata列拼接
    total_data = delete_redundant_data(unsw)

    # 特征选择
    imp_feature = ['sttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_state_ttl', 'dttl',
                   'dload', 'ackdat', 'state', 'tcprtt', 'ct_srv_dst', 'dmean', 'service',
                   'smean', 'synack', 'sbytes', 'rate', 'swin', 'dbytes',
                   'ct_src_dport_ltm', 'sload', 'ct_srv_src', 'dur', 'ct_flw_http_mthd',
                   'trans_depth', 'response_body_len', 'sloss', 'sinpkt']
    feature_data = total_data[imp_feature]
    label_data = total_data['label'].values

    # one_hot编码
    feature_data = pd.get_dummies(data=feature_data, columns=['service', 'state'])

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data = scaler.fit_transform(feature_data)

    return feature_data, label_data


# %%　加载UNSW_NB15数据集特征全集
def load_UNSW_NB15_allFeature():
    traindata = pd.read_csv('UNSW_NB15/UNSW_NB15_training-set.csv', index_col=False)
    testdata = pd.read_csv('UNSW_NB15/UNSW_NB15_testing-set.csv', index_col=False)

    unsw = pd.concat([traindata, testdata], axis=0, ignore_index=True)  # traindata与testdata列拼接
    total_data = delete_redundant_data(unsw)

    label_data = total_data['label'].values
    feature_data = total_data.drop(['id', 'attack_cat', 'label'], axis=1)

    # one_hot编码
    feature_data = pd.get_dummies(data=feature_data)

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data = scaler.fit_transform(feature_data)

    return feature_data, label_data


# %% 加载CIC-IDS2017数据集
def load_CICIDS2017():
    dataset = pd.read_csv("./CICIDS2017/dataset.csv")

    dataset[' Label'] = dataset[' Label'].apply({'DoS': 1, 'BENIGN': 0, 'DDoS': 1, 'PortScan': 1}.get)
    # data=dataset["Label"].value_counts()
    data = dataset.sample(frac=0.1)
    data = data[np.isfinite(data).all(1)]  # 去除inf和nan
    total_data = delete_redundant_data(data)

    # 特征选择
    imp_feature = [' Bwd Packet Length Std', ' Average Packet Size', ' Bwd Header Length',
                   ' Max Packet Length', 'FIN Flag Count', ' Packet Length Mean',
                   ' Destination Port', 'Bwd Packet Length Max', ' Bwd Packet Length Mean',
                   ' Avg Bwd Segment Size', ' Flow IAT Std', ' Packet Length Std',
                   ' Idle Max', ' Fwd IAT Mean', ' Min Packet Length', ' Idle Min',
                   ' Packet Length Variance', ' Fwd IAT Max', ' Fwd IAT Std',
                   ' Flow IAT Max', 'Idle Mean', ' Bwd IAT Std', ' Bwd Packet Length Min',
                   'Init_Win_bytes_forward', ' Active Std', ' Flow IAT Mean',
                   ' Flow Duration', 'Fwd IAT Total', ' URG Flag Count',
                   ' Fwd Packet Length Std', ' PSH Flag Count', ' Fwd Packet Length Min',
                   ' Total Length of Bwd Packets', ' Fwd Header Length',
                   ' Fwd Packet Length Mean', ' Init_Win_bytes_backward',
                   ' Avg Fwd Segment Size', 'Fwd PSH Flags', ' SYN Flag Count',
                   ' Bwd IAT Max', ' ACK Flag Count']

    feature_data = total_data[imp_feature]
    label_data = total_data[' Label'].values

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data = scaler.fit_transform(feature_data)

    return feature_data, label_data


# %% 加载CIC-IDS2017数据集特征全集
def load_CICIDS2017_allFeature():
    dataset = pd.read_csv("./CICIDS2017/dataset.csv")
    dataset[' Label'] = dataset[' Label'].apply({'DoS': 1, 'BENIGN': 0, 'DDoS': 1, 'PortScan': 1}.get)
    data = dataset.sample(frac=0.1)
    data = data[np.isfinite(data).all(1)]  # 去除inf和nan
    total_data = delete_redundant_data(data)

    label_data = total_data[' Label'].values
    feature_data = total_data.drop([' Label'], axis=1)

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data = scaler.fit_transform(feature_data)

    return feature_data, label_data


#

# %% 加载CIC-IDS2017数据集
def load_CICIDS2018():
    dataset = pd.read_csv("./CICIDS2018/data/02-14-2018.csv")
    # 统一去掉列名前后空格
    # dataset.columns = dataset.columns.str.strip()

    dataset['Label'] = dataset['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    # data=dataset["Label"].value_counts()

    # 剔除时间戳列，避免np.isfinite报错
    if 'Timestamp' in dataset.columns:
        dataset = dataset.drop('Timestamp', axis=1)

    data = dataset.sample(frac=0.1)
    data = data[np.isfinite(data).all(1)]  # 去除inf和nan
    total_data = delete_redundant_data(data)

    # 特征选择
    imp_feature = [' Bwd Packet Length Std', ' Average Packet Size', ' Bwd Header Length',
                   ' Max Packet Length', 'FIN Flag Count', ' Packet Length Mean',
                   ' Destination Port', 'Bwd Packet Length Max', ' Bwd Packet Length Mean',
                   ' Avg Bwd Segment Size', ' Flow IAT Std', ' Packet Length Std',
                   ' Idle Max', ' Fwd IAT Mean', ' Min Packet Length', ' Idle Min',
                   ' Packet Length Variance', ' Fwd IAT Max', ' Fwd IAT Std',
                   ' Flow IAT Max', 'Idle Mean', ' Bwd IAT Std', ' Bwd Packet Length Min',
                   'Init_Win_bytes_forward', ' Active Std', ' Flow IAT Mean',
                   ' Flow Duration', 'Fwd IAT Total', ' URG Flag Count',
                   ' Fwd Packet Length Std', ' PSH Flag Count', ' Fwd Packet Length Min',
                   ' Total Length of Bwd Packets', ' Fwd Header Length',
                   ' Fwd Packet Length Mean', ' Init_Win_bytes_backward',
                   ' Avg Fwd Segment Size', 'Fwd PSH Flags', ' SYN Flag Count',
                   ' Bwd IAT Max', ' ACK Flag Count']
    # CIC-IDS-2018 通用最优特征子集
    optimal_features_20 = [
        # --- 流量基础特征（论文共识第一梯队）---
        'Dst Port',  # 目标端口 - 两篇论文都排第一
        'Protocol',  # 协议类型
        'Flow Duration',  # 流持续时间
        'Tot Fwd Pkts',  # 正向包总数
        'Tot Bwd Pkts',  # 反向包总数
        'TotLen Fwd Pkts',  # 正向包总长度
        'TotLen Bwd Pkts',  # 反向包总长度

        # --- 包长度特征 ---
        'Pkt Len Max',  # 最大包长度
        'Pkt Len Std',  # 包长度标准差
        'Pkt Len Var',  # 包长度方差
        'Fwd Pkt Len Max',  # 正向最大包长
        'Fwd Pkt Len Min',  # 正向最小包长
        'Fwd Pkt Len Mean',  # 正向平均包长
        'Fwd Pkt Len Std',  # 正向包长标准差
        'Bwd Pkt Len Max',  # 反向最大包长
        'Bwd Pkt Len Std',  # 反向包长标准差

        # --- 时间间隔特征 ---
        'Flow IAT Mean',  # 流间隔平均值
        'Flow IAT Std',  # 流间隔标准差
        'Fwd IAT Mean',  # 正向包间隔均值
        'Fwd IAT Std',  # 正向包间隔标准差

        # --- 窗口与标志位特征 ---
        'Init Fwd Win Byts',  # 初始正向窗口大小
        'Init Bwd Win Byts',  # 初始反向窗口大小
        'Fwd Header Len',  # 正向头部长度
        'Bwd Header Len',  # 反向头部长度
    ]
    feature_data = total_data[optimal_features_20]
    label_data = total_data['Label'].values

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data = scaler.fit_transform(feature_data)

    return feature_data, label_data


# 替换你原来的 load_CICIDS2017() 函数
def load_dataset(dataset_name, dataset_path):
    """
    统一的数据集加载接口
    dataset_name: 'CICIDS2017', 'NF-BoT-IoT', 'NF-ToN-IoT'
    """
    if dataset_name == 'CICIDS2017':
        return load_CICIDS2017()
    elif dataset_name == 'NF-BoT-IoT':
        return load_NF_dataset(dataset_path)
    elif dataset_name == 'NF-ToN-IoT':
        # 使用 Nature 论文的 5 特征
        return load_NF_ToN_IoT_for_NetVax(dataset_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")



def load_NF_ToN_IoT_for_NetVax(dataset_path, feature_mode='NetVax'):

    chunks = []
    for chunk in pd.read_csv(dataset_path, chunksize=200000):
        chunks.append(chunk)
    dataset = pd.concat(chunks, ignore_index=True)
    print(f"[原始] 总样本: {len(dataset):,}")
    print(f"[原始] 标签分布:\n{dataset['Label'].value_counts()}")
    normal = dataset[dataset['Label'] == 0]
    TON_COMPACT_FEATURES = [
        'L4_DST_PORT',  # 目标端口 — 几乎所有论文排名第一
        'PROTOCOL',  # 协议类型
        'L7_PROTO',  # 应用层协议
        'FLOW_DURATION_MILLISECONDS',  # 流持续时间

        'IN_BYTES',  # 入站字节数
        'OUT_BYTES',  # 出站字节数
        'IN_PKTS',  # 入站包数
        'OUT_PKTS',  # 出站包数

        'LONGEST_FLOW_PKT',  # 最大包长
        'SHORTEST_FLOW_PKT',  # 最小包长
        'MIN_IP_PKT_LEN',  # 最小IP包长
        'MAX_IP_PKT_LEN',  # 最大IP包长

        'SRC_TO_DST_IAT_AVG',  # 正向平均包间隔
        'SRC_TO_DST_IAT_MAX',  # 正向最大包间隔
        'SRC_TO_DST_IAT_STDDEV',  # 正向包间隔标准差
        'DST_TO_SRC_IAT_AVG',  # 反向平均包间隔

        'SRC_TO_DST_SECOND_BYTES',  # 每秒入站字节
        'DST_TO_SRC_SECOND_BYTES',  # 每秒出站字节
        'RETRANSMITTED_IN_PKTS',  # 入站重传包数
        'TCP_FLAGS',  # TCP标志位汇总
    ]
    avail_features = [f for f in TON_COMPACT_FEATURES if f in dataset.columns]
    print(f"[NetVax] 使用 {len(avail_features)} 个紧凑特征: {avail_features}")

    # 删除无关列
    drop_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR',
                 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'Attack']
    dataset = dataset.drop(columns=[c for c in drop_cols if c in dataset.columns])

    # 去inf/nan
    dataset = dataset[np.isfinite(dataset).all(1)]

    # 分层采样
    _, data, _, _ = train_test_split(
        dataset, dataset['Label'],
        test_size=0.1, random_state=42, stratify=dataset['Label']
    )

    avail_features = [f for f in TON_COMPACT_FEATURES if f in data.columns]
    print(f"\n[NetVax] 使用 {len(avail_features)} 个紧凑特征")
    print(f"[NetVax] 特征列表: {avail_features}")

    feature_data = data[avail_features].values
    label_data = data['Label'].values

    print(f"[NetVax] 正常样本: {(label_data == 0).sum():,}")
    print(f"[NetVax] 攻击样本: {(label_data == 1).sum():,}")

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data = scaler.fit_transform(feature_data)

    return feature_data, label_data


def load_NF_dataset(csv_path):
    """
    读取 NF-BoT-IoT-v3 或 NF-ToN-IoT-v3 数据集
    dataset_type: 'BoT' 或 'ToN'
    """
    # 分块读取（文件太大，3.6GB / 5GB）
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=100000):
        chunks.append(chunk)
    dataset = pd.concat(chunks, ignore_index=True)

    # --- 标签处理 ---
    # Label 列: Benign=0, Attack=1
    # dataset['Label'] = dataset['Label'].apply(lambda x: 0 if str(x).strip() == 'Benign' else 1)

    # --- 删除无关列 ---
    # IP地址直接删除（或用LabelEncoder编码，但维度太高建议删除）
    cols_to_drop = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR',
                    'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
                    'Attack']  # Attack列是多分类标签，二分类时用不到
    dataset = dataset.drop(columns=[c for c in cols_to_drop if c in dataset.columns])

    # --- 去除inf和nan ---
    dataset = dataset[np.isfinite(dataset).all(1)]

    # --- 删除常量列（所有值相同）---
    nunique = dataset.nunique()
    constant_cols = nunique[nunique == 1].index.tolist()
    # 也删除几乎常量列（如99.9%相同）
    dataset = dataset.drop(columns=constant_cols)

    NF_IOT_OPTIMAL_FEATURES_20 = [
        'L4_DST_PORT',  # 目标端口 — 几乎所有论文排名第一
        'PROTOCOL',  # 协议类型
        'L7_PROTO',  # 应用层协议
        'FLOW_DURATION_MILLISECONDS',  # 流持续时间

        'IN_BYTES',  # 入站字节数
        'OUT_BYTES',  # 出站字节数
        'IN_PKTS',  # 入站包数
        'OUT_PKTS',  # 出站包数

        'LONGEST_FLOW_PKT',  # 最大包长
        'SHORTEST_FLOW_PKT',  # 最小包长
        'MIN_IP_PKT_LEN',  # 最小IP包长
        'MAX_IP_PKT_LEN',  # 最大IP包长

        'SRC_TO_DST_IAT_AVG',  # 正向平均包间隔
        'SRC_TO_DST_IAT_MAX',  # 正向最大包间隔
        'SRC_TO_DST_IAT_STDDEV',  # 正向包间隔标准差
        'DST_TO_SRC_IAT_AVG',  # 反向平均包间隔

        'SRC_TO_DST_SECOND_BYTES',  # 每秒入站字节
        'DST_TO_SRC_SECOND_BYTES',  # 每秒出站字节
        'RETRANSMITTED_IN_PKTS',  # 入站重传包数
        'TCP_FLAGS',  # TCP标志位汇总
    ]
    feature_data = dataset[NF_IOT_OPTIMAL_FEATURES_20]
    label_data = dataset['Label'].values

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data = scaler.fit_transform(feature_data)

    return feature_data, label_data


# %% ── [OPT-IO]: 数据加载缓存 ──
def load_NF_dataset_cached(dataset_name, csv_path, cache_dir='./cache'):
    """
    [OPT-IO] 带缓存的数据加载。
    首次加载 CSV 后缓存为 .npy 文件，后续直接读取，速度提升 5~10 倍。

    用法:
      feature_data, label_data = load_NF_dataset_cached(
          './NF-BoT-IoT-V3/data/NF-BoT-IoT-v3.csv'
      )
    """
    os.makedirs(cache_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    feature_cache = os.path.join(cache_dir, f'{base_name}_features.npy')
    label_cache = os.path.join(cache_dir, f'{base_name}_labels.npy')

    # 尝试读取缓存
    if os.path.exists(feature_cache) and os.path.exists(label_cache):
        print(f"[OPT-IO] 从缓存加载: {feature_cache}")
        feature_data = np.load(feature_cache)
        label_data = np.load(label_cache)
        return feature_data, label_data

    # 首次加载：调用原始加载函数
    print(f"[OPT-IO] 首次加载 CSV（较慢），后续将使用缓存...")
    feature_data, label_data = load_dataset(dataset_name, csv_path)

    # 保存缓存
    np.save(feature_cache, feature_data)
    np.save(label_cache, label_data)
    print(f"[OPT-IO] 缓存已保存: {feature_cache}")

    return feature_data, label_data
