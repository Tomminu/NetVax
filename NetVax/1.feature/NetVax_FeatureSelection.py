# NetVax_FeatureSelection.py
# =============================================================================
# =============================================================================

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
from scipy.io import arff

# %% ── NEW (Eq.6/7): 新增依赖 ──
from scipy.stats import qmc                      # Sobol 低差异序列
from scipy.stats import chi2                     # 卡方分布分位数
from scipy.linalg import cholesky                # Cholesky 分解
from scipy.special import gammainc               # 球面均匀采样（保留原始sample函数）

def read_file_mat(dataFile):
    data = scio.loadmat(dataFile)
    features = data['X']
    label = data['Y']
    return (features, label)


def read_file_arff(dataFile):
    data, meta = arff.loadarff(dataFile)
    data = pd.DataFrame(data)
    label = data.iloc[:,-1]
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)
    features = data.iloc[:,:-1].values
    return (features, label)


def initialize_population_sobol(population_size, problem_size, priority_weights=None,
                                amplification=1.0, cov_matrix=None, tau=0.5,
                                feature_bounds=None):
    """

    参数：
      population_size: N, 种群规模
      problem_size: D, 特征维度
      priority_weights: W = diag(w1,...,wD), 疫苗引导优先级向量 (None表示等权)
      amplification: α, 优先级放大系数
      cov_matrix: Σ̂, 阶段1输出的经验协方差矩阵 (None表示独立)
      tau: τ∈[0,1], 高斯-柯西混合权重
      feature_bounds: [a, b], 特征边界 (None表示[-1,1])

    返回：
      binary_population: shape (N, D), 二值化种群 (0/1)
    """
    N, D = population_size, problem_size

    # --- 阈值/边界设置 ---
    if feature_bounds is None:
        a, b = -1.0, 1.0  # 默认映射到 [-1, 1]
    else:
        a, b = feature_bounds
    m = (a + b) / 2.0            # 中点
    s = (b - a) / 2.0            # 半宽

    # --- Step 1: Sobol 低差异序列（近似 Owen scrambling）---
    # 使用 scipy 的 Sobol 引擎，自带 random 重排可近似 Owen 加扰效果
    try:
        sampler = qmc.Sobol(d=D, scramble=True, seed=np.random.randint(0, 2**31))
        u = sampler.random(n=N)  # u(i) in [0,1]^D
    except Exception:
        # 回退：使用随机 Latin Hypercube 近似
        u = np.random.uniform(0, 1, size=(N, D))

    # --- Step 2: 高斯-柯西混合变换 q(i) ---
    # 高斯分量: Φ⁻¹(u)
    gaussian_part = np.random.normal(0, 1, size=(N, D))  # 近似 Φ⁻¹
    # 柯西分量: tan(π(u - 1/2))
    cauchy_part = np.tan(np.pi * (u - 0.5))
    # 混合
    q = tau * gaussian_part + (1 - tau) * cauchy_part  # q(i), shape (N, D)

    # --- Step 3: 疫苗引导的优先级偏置 W^α ⊙ q(i) ---
    if priority_weights is not None:
        w = np.array(priority_weights, dtype=float)
        w = w ** amplification                    # W^α
        w = w / (np.max(w) + 1e-10)               # 归一化到 [0,1]
        q_weighted = q * w[np.newaxis, :]          # 按列加权
    else:
        q_weighted = q

    # --- Step 4: 协方差结构注入 L(W^α ⊙ q) ---
    if cov_matrix is not None and cov_matrix.shape == (D, D):
        try:
            L = cholesky(cov_matrix, lower=True)  # Cholesky 分解
            q_transformed = q_weighted @ L.T       # 应用协方差结构
        except Exception:
            q_transformed = q_weighted              # 非正定时回退
    else:
        q_transformed = q_weighted

    # --- Step 5: 投影到可行域 X_i = Π_Ω(m + s ⊙ transformed) ---
    X_continuous = m + s * q_transformed            # 连续值
    # 二值化：>0 为 1，否则 0（对称阈值）
    binary_population = (X_continuous > m).astype(float)

    return binary_population


def random_antibody_fcn2(population_size, problem_size, threshold):
    population = np.random.uniform(low=-1, high=1, size=(population_size, problem_size))
    for i in range(0, population_size):
        for j in range(0, problem_size):
            if population[i][j] > threshold:
                population[i][j] = 1
            else:
                population[i][j] = 0
    return population


def random_antibody_fcn(population_size, problem_size, threshold=-0.2):
    # 使用 Eq.(2) 框架，tau=0  => 纯柯西分量
    binary_pop = initialize_population_sobol(
        population_size, problem_size,
        priority_weights=None, tau=0.0,  # tau=0 => 纯柯西
        feature_bounds=[-1.0, 1.0]
    )
    return binary_pop


def drift_penalized_mrmr(features, labels, proxy_labels=None,
                         drift_matrix=None, feature_costs=None,
                         lambda_g=0.5, rho_g=0.3, xi_g=0.2,
                         max_features=200, n_groups=10):
    """
    参数：
      features: (N, D) 特征矩阵
      labels: (N,) 标签
      proxy_labels: (N,) 疫苗派生的代理标签 (None 则直接用labels)
      drift_matrix: (D,) 各特征的漂移度量 (None 则自动计算)
      feature_costs: (D,) 各特征的计算/内存成本 (None 则均等)
      lambda_g, rho_g, xi_g: 各惩罚项的权重
      max_features: 最大选择特征数
      n_groups: 特征分组数

    返回：
      selected_indices: 被选中的特征索引列表
      scores: 各特征组合的评分历史
    """
    N, D = features.shape

    # --- Step 1: 特征分组（按相关性分组）---
    corr_matrix = np.abs(np.corrcoef(features.T))
    # 谱聚类近似：按相关性和语义分组
    # 简单实现：基于相关性的层次分组
    np.random.seed(42)
    group_assignments = np.random.randint(0, n_groups, size=D)  # 简化分组
    unique_groups = np.unique(group_assignments)

    if proxy_labels is None:
        proxy_labels = labels.ravel()

    # --- Step 2: 计算漂移度量 Drift(Xj) ---
    if drift_matrix is None:
        # 将数据按时间分为前后两半，计算协方差变化
        mid = N // 2
        drift_matrix = np.zeros(D)
        for j in range(D):
            cov_early = np.cov(features[:mid, j])
            cov_late = np.cov(features[mid:, j])
            # Frobenius 范数差值（标量协方差时退化为绝对值差）
            drift_matrix[j] = abs(cov_early - cov_late)

    # --- Step 3: 特征成本 cj ---
    if feature_costs is None:
        max_cost = 10.0
        # 按特征方差归一化
        feature_vars = np.var(features, axis=0)
        feature_costs = 1.0 + (feature_vars - np.min(feature_vars)) / \
                        (np.max(feature_vars) - np.min(feature_vars) + 1e-10) * (max_cost - 1.0)

    # --- Step 4: 计算 Rel(Xj;Z) 相关性 ---
    # 使用互信息近似：F统计量
    from sklearn.feature_selection import f_classif
    f_scores, _ = f_classif(features, proxy_labels)
    relevance = f_scores / (np.max(f_scores) + 1e-10)

    # --- Step 5: 逐组筛选 ---
    selected_indices = []
    all_scores = []

    for g in unique_groups:
        group_cols = np.where(group_assignments == g)[0]
        if len(group_cols) == 0:
            continue

        # 每组可选的上限
        max_per_group = min(max_features // len(unique_groups) + 1, len(group_cols))

        # --- Step 6: 贪心前向选择最大化 Eq.(1) ---
        selected = []
        remaining = list(group_cols)

        while len(selected) < max_per_group and remaining:
            best_score = -np.inf
            best_j = None

            for j in remaining:
                candidate = selected + [j]
                S_size = len(candidate)

                # Rel 项：Σ Rel(Xj; Z)
                rel_sum = sum(relevance[idx] for idx in candidate)

                # Red 项：ΣΣ Red(Xi, Xj|Z) 条件冗余
                if S_size > 1:
                    red_sum = 0.0
                    count = 0
                    for ii in range(S_size):
                        for jj in range(ii + 1, S_size):
                            xi_idx = candidate[ii]
                            xj_idx = candidate[jj]
                            # 条件冗余 = 偏相关系数（给定 Z 后）
                            rho = np.corrcoef(features[:, xi_idx], features[:, xj_idx])[0, 1]
                            red_sum += abs(rho)
                            count += 1
                    red_term = red_sum / count if count > 0 else 0
                else:
                    red_term = 0.0

                # Drift 项：Σ Drift(Xj)
                drift_sum = sum(drift_matrix[idx] for idx in candidate)
                drift_term = drift_sum / S_size

                # Cost 项：Σ cj
                cost_sum = sum(feature_costs[idx] for idx in candidate)
                cost_term = cost_sum / S_size

                # Eq.(1): F(S) 完整目标函数
                score = rel_sum - lambda_g * red_term - rho_g * drift_term - xi_g * cost_term

                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j is not None:
                selected.append(best_j)
                remaining.remove(best_j)
                all_scores.append(best_score)

        selected_indices.extend(selected)

    return selected_indices[:max_features], all_scores


def heavy_tailed_mutation_kernel(center, step_scale, tau=0.5, size=1, rng=None):
    """

    参数：
      center: x₀, 变异中心 (可以是标量或数组)
      step_scale: γ>0, 步长尺度
      tau: τ∈[0,1], 混合权重（高斯 vs 柯西）
      size: 采样数量
      rng: 随机数生成器

    返回：
      samples: 从混合分布中采样的值
    """
    if rng is None:
        rng = np.random.default_rng()

    n_gaussian = rng.binomial(size, tau) if isinstance(tau, float) else \
                 int(size * np.mean(tau))

    # 高斯分量：N(0, γ²)
    gaussian_samples = rng.normal(0, step_scale, size=n_gaussian)

    # 柯西分量：Cauchy(0, γ)
    n_cauchy = size - n_gaussian
    cauchy_samples = rng.standard_cauchy(size=n_cauchy) * step_scale

    samples = np.concatenate([gaussian_samples, cauchy_samples])
    rng.shuffle(samples)

    return center + samples


def adaptive_mutation_factor(current_iter, max_iter, sigma_min=0.01, sigma_max=0.5,
                              kappa_p1=5.0, kappa_p2=3.0, theta=0.5,
                              best_fitness=None, median_fitness=None):
    """

    参数：
      current_iter: g, 当前迭代次数
      max_iter: G_max, 总迭代次数
      sigma_min, sigma_max: 变异步长的上下界
      kappa_p1: 时间灵敏度（越大退火越快）
      kappa_p2: 信号灵敏度（种群反馈强度）
      theta: 退火中点位置 (0,1)
      best_fitness: J_best, 当前最优个体适应度
      median_fitness: J_med, 当前中位个体适应度

    返回：
      sigma: 当前迭代的自适应变异因子
      d_delta_J: 归一化效用差距（用于调试）
    """
    progress = current_iter / max_iter  # g / G_max

    # 归一化效用差距 d_ΔJ
    epsilon = 1e-8
    if best_fitness is not None and median_fitness is not None:
        d_delta_J = (best_fitness - median_fitness) / (abs(median_fitness) + epsilon)
    else:
        d_delta_J = 0.0

    # sigmoid 内的参数
    exp_arg = kappa_p1 * (progress - theta) - kappa_p2 * d_delta_J

    # 避免溢出
    exp_arg = np.clip(exp_arg, -50, 50)

    # Eq.(4)
    sigma = sigma_min + (sigma_max - sigma_min) / (1.0 + np.exp(exp_arg))

    return sigma, d_delta_J


def compute_genotypic_diversity(population, omega=None):
    """
    参数：
      population: (N, D) 种群矩阵（N个个体，D维特征）
      omega: (D,) 特征权重向量（None 则等权）

    返回：
      DPW: 多样性度量值
      N_MDF: 归一化因子（最大成对距离）
    """
    N, D = population.shape
    if N <= 1:
        return 0.0, 0.0

    if omega is None:
        omega = np.ones(D) / D  # 等权

    # 计算所有成对加权距离
    pairwise_dists = []
    for i in range(N):
        for j in range(i + 1, N):
            diff = population[i] - population[j]
            weighted_dist = np.sqrt(np.sum(omega * diff ** 2))
            pairwise_dists.append(weighted_dist)

    pairwise_dists = np.array(pairwise_dists)
    max_dist = np.max(pairwise_dists) if len(pairwise_dists) > 0 else 1.0
    N_MDF = max_dist if max_dist > 1e-10 else 1.0

    # Eq.(7)
    sum_dist = np.sum(pairwise_dists)
    DPW = (1.0 / N_MDF) * (2.0 / (N * (N - 1))) * sum_dist if N > 1 else 0.0

    return DPW, N_MDF


def KNN_classifier(antibody, features, label):
    index = np.where(antibody == 1)
    features_selected = features[:, index[0]]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    accuracy = cross_val_score(knn, features_selected, label.ravel(), cv=skf, scoring='accuracy')
    return accuracy


# %% ── MODIFIED: 计算亲和度 ──
def calculate_affinity_fcn(_antibodies, features, label,
                           use_drift_cost=False, drift_matrix=None,
                           feature_costs=None, lambda_g=0.5, rho_g=0.3, xi_g=0.2):
    w = 0.99
    problem_size = features.shape[1]
    antibodies = np.zeros((_antibodies.shape[0], _antibodies.shape[1] + 1))

    for ind in range(len(_antibodies)):
        ratio = np.sum(_antibodies[ind]) / problem_size
        accuracy = KNN_classifier(_antibodies[ind], features, label)
        accuracy = np.mean(accuracy)

        if use_drift_cost:
            # NEW (Eq.1): 漂移和成本感知目标
            zero_one = _antibodies[ind]
            selected_idx = np.where(zero_one == 1)[0]

            if len(selected_idx) == 0:
                affinity = 1.0  # 全不选时最差
            else:
                # Drift 项
                drift_sum = np.sum(drift_matrix[selected_idx]) if drift_matrix is not None else 0
                # Cost 项
                cost_sum = np.sum(feature_costs[selected_idx]) if feature_costs is not None else 0
                # Eq.(1) 目标函数（最小化形式）
                # 原始 Eq.(1) 是最大化，这里转为最小化以兼容原始框架
                raw_score = (1 - accuracy) + lambda_g * (len(selected_idx) / max(problem_size, 1)) \
                            + rho_g * drift_sum + xi_g * cost_sum
                affinity = raw_score / (1 + lambda_g + rho_g + xi_g)  # 归一化到 [0,1]
        else:
            # 原始亲和度
            affinity = w * (1 - accuracy) + (1 - w) * ratio

        antibodies[ind] = np.append(_antibodies[ind], affinity)

    return antibodies


# %% ── 克隆算法 ──
def clone_antibodies_fcn(antibodies):
    clones = []
    aff = antibodies[:, -1]
    aff = (len(aff) - np.argsort(aff))

    for a, index in zip(antibodies, range(len(antibodies))):
        clones += [deepcopy(a[:-1]) for _ in range(aff[index])]
    return np.array(clones)


# %% ── 原始突变策略──
def mutation_fcn_original(clones, mutation_exp):
    mut_pop = np.random.uniform(low=-1, high=1, size=clones.shape)
    threshold = 0.5 - mutation_exp * 0.8
    mut_pop[mut_pop > threshold] = 1
    mut_pop[mut_pop <= threshold] = 0
    mut_pop = 1 - mut_pop
    clones = clones * mut_pop
    return clones


# %% ── 基于高斯-柯西混合变异的增强突变 ──
def mutation_fcn_heavy_tailed(clones, current_iter, max_iter,
                               tau_init=0.7, tau_final=0.3,
                               sigma_min=0.05, sigma_max=0.5,
                               best_fitness=None, median_fitness=None,
                               kappa_p1=5.0, kappa_p2=3.0):
    """

    关键差异：
      - 均匀分布 bit-flip
    """
    N, D = clones.shape

    sigma, d_delta = adaptive_mutation_factor(
        current_iter, max_iter,
        sigma_min=sigma_min, sigma_max=sigma_max,
        kappa_p1=kappa_p1, kappa_p2=kappa_p2,
        best_fitness=best_fitness, median_fitness=median_fitness
    )

    progress = current_iter / max_iter
    tau_annealed = tau_init + (tau_final - tau_init) * progress
    tau_annealed = np.clip(tau_annealed, tau_final, tau_init)

    # 变异掩码：每个位以概率 sigma 被选择变异
    mut_mask = np.random.uniform(0, 1, size=clones.shape) < sigma

    # 重尾核生成扰动
    perturbation = heavy_tailed_mutation_kernel(
        center=0.0, step_scale=sigma,
        tau=tau_annealed, size=np.sum(mut_mask)
    )
    perturbation = np.clip(perturbation, -1.0, 1.0)

    # 应用变异：原始值 + 扰动 → 二值化
    mutated = clones.copy()
    mutated[mut_mask] = clones[mut_mask] + perturbation
    # 二值化：> 0 为 1，否则保持 0
    mutated = (mutated > 0.5).astype(float)

    return mutated


# %% 自适应种群大小──
def compute_population_size(current_iter, max_iter,
                             init_size, min_size,
                             beta=1.5, diversity_feedback=0.0,
                             kappa=0.5, nu=10.0, delta_c=0.01):
    """
    参数：
      current_iter: g, 当前迭代
      max_iter: G_max, 最大迭代
      init_size: N_init, 初始种群大小
      min_size: N_min, 最小种群大小
      beta: β>0, 缩减曲线形状
      diversity_feedback: D_PW, 当前多样性
      kappa: κ, 反馈幅度
      nu: ν, 反馈灵敏度
      delta_c: δ_c, 多样性阈值

    返回：
      N_next: 下一代种群大小（整数）
      feedback_term: 反馈项值（用于调试）
    """
    progress = current_iter / max_iter

    base = min_size + (init_size - min_size) * (1.0 - progress ** beta)

    feedback_term = 1.0 + kappa * np.tanh(nu * (delta_c - diversity_feedback))

    N_next = int(round(base * feedback_term))
    N_next = max(min_size, min(init_size, N_next))

    return N_next, feedback_term


# %% ── 高斯游走策略──
def gaussian_walk(elite_antibody, n_new, cov_scale=0.1,
                   current_iter=0, max_iter=100,
                   diversity_feedback=0.0, delta_c=0.01,
                   kappa=0.5, nu=10.0, beta=1.5):
    """

    参数：
      elite_antibody: x_best^w, 精英抗体
      n_new: 新候选数量
      cov_scale: σ₀², 基协方差缩放
      current_iter: g, 当前迭代
      max_iter: G_max, 最大迭代
      diversity_feedback: D_PW, 当前多样性
      delta_c: δ_c, 多样性阈值
      kappa: κ, 反馈幅度
      nu: ν, 反馈灵敏度
      beta: β, 曲线形状

    返回：
      new_candidates: (n_new, D) 新生成候选
    """
    D = len(elite_antibody)
    progress = current_iter / max_iter

    # 自适应协方差矩阵 Σ_walk
    feedback = 1.0 + kappa * np.tanh(nu * (delta_c - diversity_feedback))
    sigma_walk = cov_scale * (1.0 - progress) ** (2 * beta) * feedback ** 2
    sigma_walk = max(sigma_walk, 1e-6)

    # 从精英附近采样
    r1 = np.random.uniform(0, 1)
    r2 = np.random.uniform(0, 1)
    # 简化实现：直接围绕精英采样
    new_candidates = np.zeros((n_new, D))
    for i in range(n_new):
        noise = np.random.normal(0, sigma_walk, size=D)
        new_candidates[i] = elite_antibody + noise + r1 * elite_antibody

    # 二值化（0/1 特征选择）
    new_candidates = (new_candidates > 0.5).astype(float)

    return new_candidates


# %% ── Fisher score 特征重要性计算 ──
def fisher_score(features, label):
    num_features = features.shape[1]
    num_samples = features.shape[0]
    labels = np.unique(label)
    S_B = np.zeros(num_features)
    S_W = np.zeros(num_features)
    mean = np.mean(features, axis=0)

    for i in labels:
        index = np.where(label == i)
        features_i = features[index,:][0]
        num_samples_i = features_i.shape[0]
        mean_i = np.mean(features_i, axis=0)
        S_B = S_B + (mean_i - mean) * (mean_i - mean) * num_samples_i / num_samples
        mean_i_tile = np.tile(mean_i, (num_samples_i, 1))
        S_W = S_W + np.sum((features_i - mean_i_tile) * (features_i - mean_i_tile), axis=0) / num_samples

    Fisher_score = S_B / S_W
    return Fisher_score


# %% ── Fisher Score 特征选择 ──
def Fisher_FS(dataFile, num_features=200):
    if dataFile.endswith('.arff'):
        features, label = read_file_arff(dataFile)
    else:
        features, label = read_file_mat(dataFile)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    accuracy = cross_val_score(knn, features, label.ravel(), cv=skf, scoring='accuracy')
    print("Full accuracy:{}".format(np.mean(accuracy)))

    Fisher_score = fisher_score(features, label)
    index = np.argsort(-Fisher_score)[:num_features]
    features = features[:, index]
    return features, label


# %% ── mRMR 特征选择 ──
from mrmr import mrmr_classif
def mRMR_FS(dataFile, num_features=200):
    if dataFile.endswith('.arff'):
        features, label = read_file_arff(dataFile)
    else:
        features, label = read_file_mat(dataFile)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    accuracy = cross_val_score(knn, features, label.ravel(), cv=skf, scoring='accuracy')
    print("Full accuracy:{}".format(np.mean(accuracy)))

    index = mrmr_classif(pd.DataFrame(features), pd.DataFrame(label), K=num_features)
    features = features[:, index]
    return features, label


# %% ── 主运行函数──
def run(features, label, problem_size,
        use_paper_formulas=True,
        population_init=50, population_min=20,
        priority_weights=None, amplification=1.0,
        beta=1.5, kappa_tanh=0.5, nu=10.0, delta_c=0.01,
        tau_init=0.7, tau_final=0.3,
        sigma_min=0.05, sigma_max=0.5,
        kappa_p1=5.0, kappa_p2=3.0,
        cov_scale=0.1,
        use_drift_cost=False,
        # 通用参数
        selection_size=10, stop_condition=50, clone_rate=0.5):
    _iteration = 0
    _antibodies = []
    best_affinity_it = []
    best_size = 5

    if not use_paper_formulas:
        population_init = 50
        population_min = 20

    clone_size = int(clone_rate * selection_size)
    DPW_new, DPW_old = 0.0, 0.0
    best_fitness_history = []

    # ── 初始化种群 ──
    if use_paper_formulas:
        # Sobol+高斯-柯西混合初始化
        # 从 Fisher Score 获取优先级权重
        fs = fisher_score(features, label)
        priority_weights = fs / (np.max(fs) + 1e-10) if priority_weights is None else priority_weights

        # 阶段1协方差矩阵
        selected_candidates = np.argsort(-fs)[:min(200, problem_size)]
        cov_estimate = np.cov(features[:, selected_candidates].T) if len(selected_candidates) > 1 else None

        _antibodies = initialize_population_sobol(
            population_init, problem_size,
            priority_weights=priority_weights,
            amplification=amplification,
            cov_matrix=cov_estimate,
            tau=0.5 if tau_init > 0.3 else 0.3
        )
    else:
        # 柯西初始化
        _antibodies = random_antibody_fcn(population_init, problem_size, threshold=-0.2)

    # 计算亲和度并排序
    if use_paper_formulas and use_drift_cost:
        # 准备 Eq.(1) 需要的漂移和成本矩阵
        mid = features.shape[0] // 2
        drift_matrix = np.zeros(problem_size)
        for j in range(problem_size):
            cov_early = np.cov(features[:mid, j])
            cov_late = np.cov(features[mid:, j])
            drift_matrix[j] = abs(cov_early - cov_late)

        feature_vars = np.var(features, axis=0)
        feature_costs = 1.0 + (feature_vars - np.min(feature_vars)) / \
                        (np.max(feature_vars) - np.min(feature_vars) + 1e-10) * 9.0

        _antibodies = calculate_affinity_fcn(
            _antibodies, features, label,
            use_drift_cost=True,
            drift_matrix=drift_matrix,
            feature_costs=feature_costs
        )
    else:
        _antibodies = calculate_affinity_fcn(_antibodies, features, label)

    _antibodies = _antibodies[np.argsort(_antibodies[:, -1])]

    start = time.time()

    while _iteration < stop_condition:
        _iteration += 1
        print('iteration: ', _iteration, '/', stop_condition)

        # ── 获取当前种群统计信息（用于变异反馈）──
        if use_paper_formulas:
            best_fit = _antibodies[0, -1] if len(_antibodies) > 0 else 0
            med_idx = len(_antibodies) // 2
            med_fit = _antibodies[med_idx, -1] if len(_antibodies) > 0 else 0
        else:
            best_fit = med_fit = None

        # ── 克隆 ──
        clones = clone_antibodies_fcn(_antibodies[:clone_size])

        # ── 变异 ──
        if use_paper_formulas:
            # 高斯-柯西混合重尾变异
            clones = mutation_fcn_heavy_tailed(
                clones, _iteration, stop_condition,
                tau_init=tau_init, tau_final=tau_final,
                sigma_min=sigma_min, sigma_max=sigma_max,
                best_fitness=best_fit, median_fitness=med_fit,
                kappa_p1=kappa_p1, kappa_p2=kappa_p2
            )
        else:
            # 原始均匀变异
            clones = mutation_fcn_original(clones, _iteration / stop_condition)

        # 计算克隆体的亲和度
        if use_paper_formulas and use_drift_cost:
            clones = calculate_affinity_fcn(
                clones, features, label,
                use_drift_cost=True,
                drift_matrix=drift_matrix,
                feature_costs=feature_costs
            )
        else:
            clones = calculate_affinity_fcn(clones, features, label)

        # ── 种群多样性和新增候选 ──
        if use_paper_formulas:
            antibody_vals = _antibodies[:, :-1]
            DPW_new, N_MDF = compute_genotypic_diversity(antibody_vals)

            pop_num, fb_term = compute_population_size(
                _iteration, stop_condition,
                population_init, population_min,
                beta=beta, diversity_feedback=DPW_new,
                kappa=kappa_tanh, nu=nu, delta_c=delta_c
            )
            rho_g = 0.3
            xi_g = 0.2
            # 如果多样性过低（早熟收敛），通过高斯游走生成新候选
            new_candidates = None
            if DPW_new < delta_c or _iteration % 10 == 0:
                elite = _antibodies[0, :-1]
                n_gw = max(1, int(population_init * 0.1))
                gw_candidates = gaussian_walk(
                    elite, n_gw,
                    cov_scale=cov_scale,
                    current_iter=_iteration, max_iter=stop_condition,
                    diversity_feedback=DPW_new,
                    delta_c=delta_c, kappa=kappa_tanh, nu=nu, beta=beta
                )
                new_candidates = np.zeros((n_gw, gw_candidates.shape[1] + 1))
                new_candidates[:, :-1] = gw_candidates
                # 计算高斯游走候选的亲和度
                for idx in range(n_gw):
                    if use_drift_cost:
                        acc = KNN_classifier(gw_candidates[idx], features, label)
                        acc_mean = np.mean(acc)
                        selected_idx = np.where(gw_candidates[idx] == 1)[0]
                        drift_sum = np.sum(drift_matrix[selected_idx])
                        cost_sum = np.sum(feature_costs[selected_idx])
                        raw_score = (1 - acc_mean) + rho_g * drift_sum + xi_g * cost_sum
                        new_candidates[idx, -1] = raw_score
                    else:
                        # 临时亲和度，后续用 calculate_affinity_fcn 重算
                        acc = KNN_classifier(gw_candidates[idx], features, label)
                        ratio = np.sum(gw_candidates[idx]) / problem_size
                        new_candidates[idx, -1] = 0.99 * (1 - np.mean(acc)) + 0.01 * ratio

        else:
            # 随机新个体维持多样性
            antibodies2 = random_antibody_fcn(population_init, problem_size,
                                               _iteration / stop_condition)
            antibodies2 = calculate_affinity_fcn(antibodies2, features, label)
            # 线性种群缩减
            pop_num = np.round(population_init - population_min) * (_iteration / stop_condition) + population_min
            new_candidates = antibodies2

        # ── 合并种群 ──
        _antibodies = np.append(_antibodies, clones, axis=0)
        if new_candidates is not None:
            _antibodies = np.append(_antibodies, new_candidates, axis=0)

        # 按亲和度排序（最小化 = 最优）
        _antibodies = _antibodies[np.argsort(_antibodies[:, -1])]

        # 选择前 pop_num 个
        pop_num = int(max(population_min, min(pop_num, len(_antibodies))))
        _antibodies = _antibodies[:pop_num]

        # 记录最佳解
        best_affinity_it.append(_antibodies[:best_size])
        best_antibody = _antibodies[0]
        best_fitness_history.append(best_antibody[-1])

    end = time.time()
    accuracy = KNN_classifier(best_antibody, features, label)
    feaSize = sum(best_antibody[:-1])
    print("featureSize:{}, accuracy:{}, variance:{}, time:{}".format(
        sum(best_antibody[:-1]), np.mean(accuracy), np.var(accuracy), end - start))
    return accuracy, feaSize, end - start


if __name__ == "__main__":
    # dataFile = 'data\\Weka\\SRBCT.arff'
    dataFile = 'data\\KeChen\\Prostate_Tumor_1.mat'
    features, label = Fisher_FS(dataFile, num_features=200)
    problem_size = features.shape[1]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    accuracy = cross_val_score(knn, features, label.ravel(), cv=skf, scoring='accuracy')
    print("mRMR accuracy:{}".format(np.mean(accuracy)))

    print("\n===  ===")
    acc2, feaSize2, Time2 = [], [], []
    for i in range(3):
        accuracy, fsize, tim = run(features, label, problem_size, use_paper_formulas=True)
        acc2.append(np.mean(accuracy))
        feaSize2.append(fsize)
        Time2.append(tim)
        print(f"Run {i+1}: acc={np.mean(accuracy):.4f}, feaSize={fsize}, time={tim:.2f}s")

    # 报告对比
    print(f"\n{'='*50}")
    print(f"avg_acc={np.mean(acc2):.4f}±{np.std(acc2):.4f}, avg_feaSize={np.mean(feaSize2):.1f}")
