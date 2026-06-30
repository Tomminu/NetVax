# NetVax_Model_Optimized.py
# =============================================================================
# 性能优化版本
# =============================================================================

# %% ── 导入 ──
import numpy as np
import pandas as pd
import torch
import os
import time
from torchmetrics.functional import pairwise_euclidean_distance
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.special import gammainc
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from scipy.optimize import minimize
from scipy.linalg import inv
from collections import deque
import gc

# 自动检测设备，优先使用 CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {DEVICE}")

# 显存分块计算默认 chunk 大小（增大减少 Python 循环次数）
GPU_CHUNK_SIZE = 20000


def clear_gpu_memory(full=False):
    """
    参数:
      full: True = 完整清理(含 empty_cache)，仅在重大阶段转换时调用。
            False = 轻量清理(仅 sync + gc)，在计算循环中安全使用。
    注意: 高频调用 empty_cache 会严重拖慢性能（强迫 CUDA 每次重分配内存）。
    """
    # if DEVICE.type == 'cuda':
    #     torch.cuda.synchronize()
    #     if full:
    #         torch.cuda.empty_cache()
    # gc.collect()


# %% ── 训练部分 ──
def getAccuracy(predict_labels, labels):
    accuracy = accuracy_score(labels, predict_labels) * 100.0
    p = precision_score(labels, predict_labels, average='macro', zero_division=0) * 100.0
    r = recall_score(labels, predict_labels, average='macro', zero_division=0) * 100.0
    f1score = f1_score(labels, predict_labels, average='macro') * 100.0
    print(" Accuracy=%.2f%% precision=%.3f%%, recall=%.3f%%, f1score=%.3f%%"
          % (accuracy, p, r, f1score))
    return (accuracy, p, r, f1score)


# %% ── 马氏距离度量 ──
# 矩阵运算 + GPU 加速
class MahalanobisMetric:
   
    def __init__(self, cov_matrix=None, reg=1e-6, device=None):
        """
        参数：
          cov_matrix: Σ̂_k, 选定k维子空间的协方差矩阵
          reg: 正则化系数，确保SPD（半正定）
          device:  torch 设备
        """
        self.cov_matrix = cov_matrix
        self.cov_inv = None
        self.reg = reg
        #  记录设备
        self.device = device if device is not None else DEVICE

        # [OPT-DATA] Y 缓存：避免在 Calculate_Radius 循环中重复 CPU→GPU 传输和 Y_cov 计算
        self._Y_cache_key = None  # id(Y_numpy) 作为缓存键
        self._Y_cache_t = None  # Y 的 GPU 张量
        self._Y_cache_cov = None  # YΣ⁻¹
        self._Y_cache_YY = None  # diag(YΣ⁻¹Yᵀ)

    def fit(self, data):
        """从数据拟合协方差矩阵并计算其逆。"""
        if data.shape[0] < 2:
            self.cov_matrix = np.eye(data.shape[1])
            self.cov_inv = self.cov_matrix
            return

        self.cov_matrix = np.cov(data.T)
        self.cov_matrix += self.reg * np.eye(self.cov_matrix.shape[0])
        try:
            self.cov_inv = inv(self.cov_matrix)
        except np.linalg.LinAlgError:
            self.cov_inv = np.linalg.pinv(self.cov_matrix)

        #  预计算协方差逆矩阵的 GPU 张量，避免每次距离计算重复传输
        self._cov_inv_t = torch.tensor(
            self.cov_inv, dtype=torch.float32, device=self.device
        )
        # [OPT-DATA] 协方差更新后，旧 Y 缓存失效
        self._Y_cache_key = None
        self._Y_cache_t = None
        self._Y_cache_cov = None
        self._Y_cache_YY = None

    def distance(self, x, y):
        """单对马氏距离（保持不变，用于少量计算场景）。"""
        if self.cov_inv is None:
            return np.sqrt(np.sum((x - y) ** 2))
        diff = x - y
        return np.sqrt(np.dot(np.dot(diff, self.cov_inv), diff))

    #  全新方法: 批量向量化马氏距离
    @torch.no_grad()
    def batch_distance(self, X, Y):
        """

        返回矩阵 D，其中 D[i,j] = d(X[i], Y[j])

        注意: 当 Y 规模很大时(>10000)，请使用 batch_distance_chunked 避免 OOM。
        """
        if self.cov_inv is None:
            #  即使回退欧氏距离也走 GPU
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
            return pairwise_euclidean_distance(X_t, Y_t).cpu().numpy()

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)

        X_cov = X_t @ self._cov_inv_t
        Y_cov = Y_t @ self._cov_inv_t

        XX = torch.sum(X_cov * X_t, dim=1, keepdim=True)  # (N, 1)
        YY = torch.sum(Y_cov * Y_t, dim=1, keepdim=True)  # (M, 1)
        XY = X_cov @ Y_t.T

        D = torch.sqrt(torch.clamp(XX - 2 * XY + YY.T, min=0.0))
        return D.cpu().numpy()

    @torch.no_grad()
    def batch_distance_chunked(self, X, Y, chunk_size=None, reduction='none'):
        """
         分块计算马氏距离，永远不物化完整的 N×M 矩阵。

        参数:
          X: (N, D) 查询点
          Y: (M, D) 参考点
          chunk_size: 每块处理的 Y 行数，默认 GPU_CHUNK_SIZE
          reduction: 'none' | 'min' | 'max'
            - 'none': 返回完整 (N, M) 矩阵（仅适合小数据）
            - 'min':  返回每行的最小值，向量 (N,) — 用于 Gen_Self_Detector
            - 'max':  返回每行的最大值，向量 (N,)

        Y 缓存：在 Calculate_Radius 等循环中，Abnormaldata 对象不变，
        首次调用时缓存 Y_t/Y_cov/YY，后续调用直接复用 GPU 张量，
        省去大量 CPU→GPU 传输和矩阵乘法。
        """
        if chunk_size is None:
            chunk_size = GPU_CHUNK_SIZE

        if self.cov_inv is None:
            # 回退到欧氏距离分块
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            result = None
            for j in range(0, len(Y), chunk_size):
                Y_chunk = Y[j:j + chunk_size]
                Y_t = torch.tensor(Y_chunk, dtype=torch.float32, device=self.device)
                dist_chunk = pairwise_euclidean_distance(X_t, Y_t)  # (N, chunk)

                if reduction == 'min':
                    chunk_val = dist_chunk.min(dim=1, keepdim=False).values
                    result = chunk_val if result is None else torch.minimum(result, chunk_val)
                elif reduction == 'max':
                    chunk_val = dist_chunk.max(dim=1, keepdim=False).values
                    result = chunk_val if result is None else torch.maximum(result, chunk_val)
                else:
                    result = dist_chunk if result is None else torch.cat([result, dist_chunk], dim=1)
                del Y_t, dist_chunk

            del X_t
            result_cpu = result.cpu().numpy() if result is not None else None
            del result
            clear_gpu_memory(full=False)
            return result_cpu

        Y_key = id(Y) if isinstance(Y, np.ndarray) else None
        if Y_key is not None and Y_key == self._Y_cache_key:
            Y_t, Y_cov_full, YY_full = (
                self._Y_cache_t, self._Y_cache_cov, self._Y_cache_YY
            )
            use_cache = True
        else:
            Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
            Y_cov_full = Y_t @ self._cov_inv_t  # (M, k)
            YY_full = torch.sum(Y_cov_full * Y_t, dim=1, keepdim=True)  # (M, 1)
            if Y_key is not None:
                self._Y_cache_key = Y_key
                self._Y_cache_t = Y_t
                self._Y_cache_cov = Y_cov_full
                self._Y_cache_YY = YY_full
            use_cache = False

        # 马氏距离分块（使用缓存张量的视图）
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_cov = X_t @ self._cov_inv_t  # (N, k)
        XX = torch.sum(X_cov * X_t, dim=1, keepdim=True)  # (N, 1)

        result = None
        for j in range(0, len(Y), chunk_size):
            j_end = min(j + chunk_size, len(Y))

            Y_cov = Y_cov_full[j:j_end]  # (chunk, k) — 仅用于 YY 预计算
            YY = YY_full[j:j_end]  # (chunk, 1)
            Y_t_chunk = Y_t[j:j_end]  # 原始 Y 切片

            XY = X_cov @ Y_t_chunk.T  # (X·Σ⁻¹) @ Y^T = X·Σ⁻¹·Y^T  ← 正确

            D_chunk = torch.sqrt(torch.clamp(XX - 2 * XY + YY.T, min=0.0))

            if reduction == 'min':
                chunk_val = D_chunk.min(dim=1, keepdim=False).values
                result = chunk_val if result is None else torch.minimum(result, chunk_val)
            elif reduction == 'max':
                chunk_val = D_chunk.max(dim=1, keepdim=False).values
                result = chunk_val if result is None else torch.maximum(result, chunk_val)
            else:
                result = D_chunk if result is None else torch.cat([result, D_chunk], dim=1)

            del XY, D_chunk

        del X_t, X_cov, XX, Y_cov, YY
        result_cpu = result.cpu().numpy() if result is not None else None
        del result
        clear_gpu_memory(full=False)
        return result_cpu


# %% ──自体检测器进化耐受──
def Gen_Self_Detector_Euclidean(Normaldata, Abnormaldata, Nonself_Radius):
    process_Nor = torch.tensor(Normaldata)
    process_Abnor = torch.tensor(Abnormaldata)
    Self_Nonself_distance = pairwise_euclidean_distance(process_Nor, process_Abnor)
    Radius = torch.min(Self_Nonself_distance, 1)
    return Radius


# %% ── 几何中位数 ──
def geometric_median(points, max_iter=100, tol=1e-8):
    c = np.median(points, axis=0)
    for iteration in range(max_iter):
        dists = np.sqrt(np.sum((points - c) ** 2, axis=1))
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists
        c_new = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)
        if np.sum((c_new - c) ** 2) < tol:
            break
        c = c_new
    return c


# %% ── 疫苗原型 ──
def vaccine_prototype(samples, k_dim, alpha=0.95):
    if len(samples) == 0:
        return None
    c_z = geometric_median(samples)
    r_z = np.sqrt(chi2.ppf(alpha, df=k_dim))
    distances = [np.sqrt(np.sum((s - c_z) ** 2)) for s in samples]
    r_z = max(r_z, np.percentile(distances, alpha * 100) if len(distances) > 0 else r_z)
    return {'center': c_z, 'radius': r_z, 'type': None}


# %% ── 疫苗库 ──
class VaccineLibrary:
    def __init__(self, k_dim, alpha=0.95, metric=None):
        self.k_dim = k_dim
        self.alpha = alpha
        self.metric = metric if metric is not None else MahalanobisMetric()
        self.self_vaccines = []
        self.nonself_vaccines = []

    def add_self_vaccine(self, samples):
        proto = vaccine_prototype(samples, self.k_dim, self.alpha)
        if proto:
            proto['type'] = 'self'
            self.self_vaccines.append(proto)
        return proto

    def add_nonself_vaccine(self, samples):
        proto = vaccine_prototype(samples, self.k_dim, self.alpha)
        if proto:
            proto['type'] = 'nonself'
            self.nonself_vaccines.append(proto)
        return proto

    def get_all_vaccines(self):
        return self.self_vaccines + self.nonself_vaccines

    def get_self_centers(self):
        return np.array([v['center'] for v in self.self_vaccines]) if self.self_vaccines else np.array([])

    def get_nonself_centers(self):
        return np.array([v['center'] for v in self.nonself_vaccines]) if self.nonself_vaccines else np.array([])


# %% ── : 新检测器实例化条件检查 ──
def check_detector_instantiation(candidate_center, candidate_radius,
                                 existing_detectors, self_vaccines,
                                 recent_flows=None, metric=None):
    if metric is None:
        metric = MahalanobisMetric()

    #  条件(1a): 批量检查新覆盖
    provides_new_coverage = True
    if recent_flows is not None and len(recent_flows) > 0 and len(existing_detectors) > 0:
        recent_arr = np.array(recent_flows)
        cand_center = np.asarray(candidate_center).reshape(1, -1)

        # 到候选检测器的距离（批量）
        d_to_candidate = metric.batch_distance(recent_arr, cand_center).ravel()
        covered_by_candidate = d_to_candidate <= candidate_radius

        if np.any(covered_by_candidate):
            # 这些被候选覆盖的点，检查是否已被现有检测器覆盖
            covered_points = recent_arr[covered_by_candidate]
            existing_centers = np.array([ec for ec, er in existing_detectors])
            existing_radii = np.array([er for ec, er in existing_detectors])

            d_to_existing = metric.batch_distance(covered_points, existing_centers)
            already_covered = (d_to_existing <= existing_radii[np.newaxis, :]).any(axis=1)
            uncovered_count = np.sum(~already_covered)
            provides_new_coverage = uncovered_count > 0
        else:
            provides_new_coverage = False

    if not provides_new_coverage:
        return False, "No new coverage provided by this candidate"

    # 条件(2): 与所有自体疫苗的安全间隔
    for sv in self_vaccines:
        d = metric.distance(candidate_center, sv['center'])
        if d < candidate_radius + sv['radius']:
            return False, f"Conflict with self-vaccine: d={d:.4f} < r_cand+r_self={candidate_radius + sv['radius']:.4f}"

    return True, "Clean"


# %% ── 冲突感知几何投影 ──
def conflict_aware_projection(detector_center, detector_radius,
                              self_vaccines, nonself_vaccines=None,
                              r_min=0.01, r_max=10.0, metric=None):
    if metric is None:
        metric = MahalanobisMetric()

    k_dim = len(detector_center)
    x0 = np.concatenate([detector_center, [detector_radius]])
    constraints = []

    for sv in self_vaccines:
        def make_self_constraint(sv_center, sv_radius):
            def constraint(x):
                c = x[:-1]
                r = x[-1]
                d = metric.distance(c, sv_center)
                return r - d + sv_radius

            return constraint

        constraints.append({
            'type': 'ineq',
            'fun': make_self_constraint(sv['center'], sv['radius'])
        })

    if nonself_vaccines is not None:
        for nv in nonself_vaccines:
            def make_nonself_constraint(nv_center, nv_radius):
                def constraint(x):
                    c = x[:-1]
                    r = x[-1]
                    d = metric.distance(c, nv_center)
                    return d - r + nv_radius

                return constraint

            constraints.append({
                'type': 'ineq',
                'fun': make_nonself_constraint(nv['center'], nv['radius'])
            })

    bounds = [(None, None)] * k_dim + [(r_min, r_max)]

    def objective(x):
        return np.sum((x - x0) ** 2)

    try:
        result = minimize(objective, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 100, 'ftol': 1e-8})
        if result.success:
            c_new = result.x[:-1]
            r_new = max(r_min, min(r_max, result.x[-1]))
            return c_new, r_new, True
        else:
            return _simple_adjustment(detector_center, detector_radius,
                                      self_vaccines, nonself_vaccines,
                                      metric, r_min, r_max)
    except Exception:
        return _simple_adjustment(detector_center, detector_radius,
                                  self_vaccines, nonself_vaccines,
                                  metric, r_min, r_max)


def _simple_adjustment(center, radius, self_vaccines, nonself_vaccines,
                       metric, r_min, r_max):
    r = radius
    for sv in self_vaccines:
        d = metric.distance(center, sv['center'])
        if d < r + sv['radius']:
            r = max(r_min, d - sv['radius'] - 1e-6)
    if nonself_vaccines is not None:
        for nv in nonself_vaccines:
            d = metric.distance(center, nv['center'])
            if d > r - nv['radius']:
                r = max(r, d + nv['radius'] + 1e-6)
    r = np.clip(r, r_min, r_max)
    return center, r, False


# %% ── 受控遗忘机制 ──
class ControlledForgetting:

    def __init__(self, window_size=1000, tau_del=0.1, T_obsolete=50,
                 T_mature=20, alpha=0.5, beta=0.3, epsilon=1e-6):
        self.window_size = window_size
        self.tau_del = tau_del
        self.T_obsolete = T_obsolete
        self.T_mature = T_mature
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.detectors = {}
        self.next_id = 0

    def add_detector(self, center, radius):
        det_id = self.next_id
        self.next_id += 1
        self.detectors[det_id] = {
            'center': center.copy(),
            'radius': radius,
            'S': 0.5,
            'age': 0,
            'b_window': deque(maxlen=self.window_size),
            'c_window': deque(maxlen=self.window_size),
            'birth_time': time.time()
        }
        return det_id

    def record_validation(self, det_id, is_conflict=False):
        if det_id not in self.detectors:
            return
        det = self.detectors[det_id]
        if is_conflict:
            det['c_window'].append(1)
            det['b_window'].append(0)
        else:
            det['b_window'].append(1)
            det['c_window'].append(0)
        det['age'] += 1

    def compute_pruning_score(self, det_id):
        if det_id not in self.detectors:
            return 1.0
        det = self.detectors[det_id]
        b_i = sum(det['b_window'])
        c_i = sum(det['c_window'])
        conflict_ratio = c_i / (b_i + c_i + self.epsilon)
        is_obsolete = 1.0 if det['age'] >= self.T_obsolete else 0.0
        h = self.alpha * conflict_ratio + self.beta * is_obsolete
        return np.clip(h, 0.0, 1.0)

    def update_memory_strength(self, det_id):
        if det_id not in self.detectors:
            return 0.0
        det = self.detectors[det_id]
        h = self.compute_pruning_score(det_id)
        exp_h = np.exp(-h)
        b_i = sum(det['b_window'])
        c_i = sum(det['c_window'])
        consistency = b_i / (b_i + c_i + self.epsilon)
        det['S'] = exp_h * det['S'] + (1 - exp_h) * consistency
        return det['S']

    def get_dead_detectors(self):
        dead_ids = []
        for det_id, det in self.detectors.items():
            if det['S'] <= self.tau_del and det['age'] >= self.T_mature:
                dead_ids.append(det_id)
        return dead_ids

    def prune(self):
        dead_ids = self.get_dead_detectors()
        for det_id in dead_ids:
            del self.detectors[det_id]
        return len(dead_ids)

    def get_alive_detectors(self):
        return [(det['center'], det['radius'])
                for det_id, det in self.detectors.items()
                if not (det['S'] <= self.tau_del and det['age'] >= self.T_mature)]

    def get_detector_count(self):
        return len([d for d in self.detectors if
                    not (self.detectors[d]['S'] <= self.tau_del and
                         self.detectors[d]['age'] >= self.T_mature)])


# %% ── : 完整检测器封装 ──
class NetVaxDetector:

    def __init__(self, k_dim, budget=100, alpha=0.95,
                 r_min=0.01, r_max=10.0,
                 tau_del=0.1, T_obsolete=50, T_mature=20,
                 window_size=1000):
        self.k_dim = k_dim
        self.budget = budget
        #  马氏距离度量自动使用 GPU
        self.metric = MahalanobisMetric(device=DEVICE)
        self.vaccines = VaccineLibrary(k_dim, alpha, self.metric)
        self.forgetting = ControlledForgetting(
            window_size=window_size, tau_del=tau_del,
            T_obsolete=T_obsolete, T_mature=T_mature
        )
        self.r_min = r_min
        self.r_max = r_max
        self.total_flows_processed = 0

    def fit_metric(self, data):
        self.metric.fit(data)

    def add_self_vaccine(self, samples):
        return self.vaccines.add_self_vaccine(samples)

    def add_nonself_vaccine(self, samples):
        return self.vaccines.add_nonself_vaccine(samples)

    def instantiate_detector(self, candidate_center, candidate_radius,
                             recent_flows=None):
        existing = [(det['center'], det['radius'])
                    for det in self.forgetting.detectors.values()]
        should_instantiate, reason = check_detector_instantiation(
            candidate_center, candidate_radius,
            existing, self.vaccines.self_vaccines,
            recent_flows, self.metric
        )
        if not should_instantiate:
            return None, reason
        c_proj, r_proj, converged = conflict_aware_projection(
            candidate_center, candidate_radius,
            self.vaccines.self_vaccines,
            self.vaccines.nonself_vaccines,
            self.r_min, self.r_max, self.metric
        )
        if self.forgetting.get_detector_count() >= self.budget:
            pruned = self.forgetting.prune()
            if self.forgetting.get_detector_count() >= self.budget:
                return None, "Budget exceeded even after pruning"
        det_id = self.forgetting.add_detector(c_proj, r_proj)
        return det_id, f"Instantiated (projection={'converged' if converged else 'fallback'})"

    def classify(self, flow):
        alive = self.forgetting.get_alive_detectors()
        if not alive:
            return 1
        for center, radius in alive:
            d = self.metric.distance(flow, center)
            if d <= radius:
                return 0
        return 1

    def classify_batch(self, flows):
        return np.array([self.classify(f) for f in flows])

    def process_flow(self, flow, true_label=None, is_conflict=None):
        self.total_flows_processed += 1
        label = self.classify(flow)
        if is_conflict is not None:
            for det_id in self.forgetting.detectors:
                self.forgetting.record_validation(det_id, is_conflict)
        elif true_label is not None:
            for det_id, det in self.forgetting.detectors.items():
                d = self.metric.distance(flow, det['center'])
                is_in = d <= det['radius']
                if is_in and true_label == 1:
                    self.forgetting.record_validation(det_id, is_conflict=True)
                elif is_in and true_label == 0:
                    self.forgetting.record_validation(det_id, is_conflict=False)
        for det_id in list(self.forgetting.detectors.keys()):
            self.forgetting.update_memory_strength(det_id)
        if self.total_flows_processed % 100 == 0:
            self.forgetting.prune()
        return label


# %% ── 超球体均匀采样 ──
def sample(center, radius, n_per_sphere):
    r = radius
    ndim = center.size
    x = np.random.normal(size=(n_per_sphere, ndim))
    ssq = np.sum(x ** 2, axis=1)
    fr = r * gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n_per_sphere, 1), (1, ndim))
    p = center + np.multiply(x, frtiled)
    return p


# %% ──  +  + : 马氏距离版 Gen_Self_Detector ──
@torch.no_grad()
def Gen_Self_Detector(Normaldata, Abnormaldata, Nonself_Radius,
                      use_mahalanobis=False, metric=None):
    if use_mahalanobis and metric is not None:
        min_distances = metric.batch_distance_chunked(
            Normaldata, Abnormaldata, reduction='min'
        )
        return min_distances
    else:
        process_Nor = torch.tensor(Normaldata, dtype=torch.float32, device=DEVICE)
        result = None
        for j in range(0, len(Abnormaldata), GPU_CHUNK_SIZE):
            chunk = Abnormaldata[j:j + GPU_CHUNK_SIZE]
            process_Abnor = torch.tensor(chunk, dtype=torch.float32, device=DEVICE)
            dist = pairwise_euclidean_distance(process_Nor, process_Abnor)
            result = dist if result is None else torch.minimum(result, dist)
            del process_Abnor, dist

        Radius = result if result is not None else torch.zeros(len(process_Nor))
        del process_Nor
        result_cpu = Radius.cpu().numpy() if hasattr(Radius, 'cpu') else Radius.numpy()
        del Radius, result
        clear_gpu_memory()
        return result_cpu


# %% ── : 计算半径 ──
def Calculate_Radius(step, Normaldata, Abnormaldata,
                     use_mahalanobis=False, metric=None):
    nonself_radius = 0
    Normaldata_num = len(Normaldata)
    Radius = np.empty(shape=Normaldata_num)

    for i in range(0, Normaldata_num, step):
        print("实验进行{}/{}.".format(
            i + step if i + step < Normaldata_num else Normaldata_num, Normaldata_num))
        local_Normaldata = Normaldata[i:i + step, :]
        curr_radius = Gen_Self_Detector(local_Normaldata, Abnormaldata, nonself_radius,
                                        use_mahalanobis=use_mahalanobis, metric=metric)
        Radius[i:i + step] = np.array(curr_radius * 0.5)

    Radius = Radius.T
    return Radius


# %% ── : 应激理论调整 ──
def tune_mature(step, mature_detector, vacc_NonSelf,
                use_mahalanobis=False, metric=None):
    for i in range(0, len(mature_detector), step):
        print("应激实验进行{}/{}.".format(
            i + step if i + step < len(mature_detector) else len(mature_detector),
            len(mature_detector)))
        local_detector = mature_detector[i:i + step, :]

        if use_mahalanobis and metric is not None:
            temp_radius = Gen_Self_Detector(local_detector[:, :-1], vacc_NonSelf, 0,
                                            use_mahalanobis=True, metric=metric)
        else:
            temp_radius = Gen_Self_Detector(local_detector[:, :-1], vacc_NonSelf, 0)

        temp_radius = np.array(temp_radius * 0.5)
        curr_radius = local_detector[:, -1]
        mature_detector[i:i + step, -1] = np.minimum(temp_radius, curr_radius)

    return mature_detector


# %% ── 检测器成熟耐受 ──
@torch.no_grad()
def detectorsTolerate(detector):
    detector = detector[np.lexsort(-detector.T)]
    mature_detector = detector[0, :].reshape((1, -1))
    mature_detector_c_t = None  #  延迟初始化，在需要时创建

    for i in range(1, len(detector)):
        print("process {}/{}".format(i, len(detector)))
        curr_center = detector[i, :-1].reshape((1, -1))
        curr_radius = detector[i, -1]
        mature_detector_c = mature_detector[:, :-1]
        mature_detector_r = mature_detector[:, -1]

        #  欧氏距离计算也移到 GPU
        curr_center_t = torch.tensor(curr_center, dtype=torch.float32, device=DEVICE)
        mature_detector_c_t = torch.tensor(mature_detector_c, dtype=torch.float32, device=DEVICE)
        dis = pairwise_euclidean_distance(curr_center_t, mature_detector_c_t).cpu().numpy()

        tmp = np.tile(curr_radius, (1, len(mature_detector_r)))
        r1_add_r2 = tmp + mature_detector_r
        r1_sub_r2 = mature_detector_r - tmp

        if np.any(np.array(dis) < r1_sub_r2):
            continue
        if np.all(np.array(dis) > r1_add_r2):
            mature_detector = np.r_[mature_detector, detector[i, :].reshape((1, -1))]
            continue

        #  降低采样点数，减少显存占用
        n_per_sphere = 500
        p = sample(curr_center, curr_radius, n_per_sphere)

        #  分块计算采样点到成熟检测器的距离
        p_t = torch.tensor(p, dtype=torch.float32, device=DEVICE)
        total_recovered_points = 0
        for j in range(0, len(mature_detector_c), GPU_CHUNK_SIZE):
            chunk_end = min(j + GPU_CHUNK_SIZE, len(mature_detector_c))
            mc_chunk_t = mature_detector_c_t[j:chunk_end]
            mr_chunk = mature_detector_r[j:chunk_end]

            dist_chunk = pairwise_euclidean_distance(p_t, mc_chunk_t)
            mr_tile = torch.tensor(
                np.tile(mr_chunk, (n_per_sphere, 1)),
                dtype=torch.float32, device=DEVICE
            )
            covered_in_chunk = (dist_chunk < mr_tile).any(dim=1)
            total_recovered_points += covered_in_chunk.sum().item()
            del mc_chunk_t, mr_tile, dist_chunk, covered_in_chunk

        del p_t
        clear_gpu_memory()

        if total_recovered_points < n_per_sphere * 0.9:
            mature_detector = np.r_[mature_detector, detector[i].reshape((1, -1))]

    return mature_detector


# %% ──  +  + : 测试分类 ──
@torch.no_grad()
def selfSampleFun(test_sample, detector_center, detector_radius_m,
                  use_mahalanobis=False, metric=None):
    n_test = len(test_sample)

    if use_mahalanobis and metric is not None:
        covered = torch.zeros(n_test, dtype=torch.bool, device=DEVICE)

        for j in range(0, len(detector_center), GPU_CHUNK_SIZE):
            center_chunk = detector_center[j:j + GPU_CHUNK_SIZE]
            radius_chunk = detector_radius_m[:, j:j + GPU_CHUNK_SIZE]

            distances = metric.batch_distance_chunked(
                test_sample, center_chunk, reduction='none'
            )
            dist_t = torch.tensor(distances, dtype=torch.float32, device=DEVICE)
            radius_t = torch.tensor(radius_chunk, dtype=torch.float32, device=DEVICE)
            covered |= (dist_t <= radius_t).any(dim=1)
            del dist_t, radius_t, distances

        mask = covered.cpu().numpy()
        del covered
        clear_gpu_memory(full=False)
        local_predict_label = mask.astype(int) ^ 1
    else:
        #  分块欧氏距离
        test_t = torch.tensor(test_sample, dtype=torch.float32, device=DEVICE)
        covered = torch.zeros(n_test, dtype=torch.bool, device=DEVICE)

        for j in range(0, len(detector_center), GPU_CHUNK_SIZE):
            center_t = torch.tensor(
                detector_center[j:j + GPU_CHUNK_SIZE],
                dtype=torch.float32, device=DEVICE
            )
            distance = pairwise_euclidean_distance(test_t, center_t)
            distance[torch.isnan(distance)] = 0
            radius_t = torch.tensor(
                detector_radius_m[:, j:j + GPU_CHUNK_SIZE],
                dtype=torch.float32, device=DEVICE
            )
            covered |= (distance <= radius_t).any(dim=1)
            del center_t, distance, radius_t

        mask = covered.cpu().numpy()
        del test_t, covered
        clear_gpu_memory(full=False)
        local_predict_label = mask.astype(int) ^ 1

    return local_predict_label


def detect(step, Radius, Center, test_Sample,
           use_mahalanobis=False, metric=None):
    num = len(test_Sample)
    detector_radius_m = np.tile(Radius, (step, 1))
    predict_labels = np.zeros(num)

    for i in range(0, num, step):
        if i + step > num:
            detector_radius_m = np.tile(Radius, (num - i, 1))
        print("测试进行{}/{}.".format(i + step if i + step < num else num, num))
        local_predict_label = selfSampleFun(
            test_Sample[i:i + step, :], Center, detector_radius_m,
            use_mahalanobis=use_mahalanobis, metric=metric
        )
        predict_labels[i:i + step] = local_predict_label

    return predict_labels


# %% ──  + : 完整流水线 ──
def run_netvax_pipeline(feature_data, label_data,
                        num_vaccines=5, step=800,
                        use_mahalanobis=True,
                        use_controlled_forgetting=True,
                        k_dim=None,
                        subsample_ratio=1.0):
    """
    参数:
      subsample_ratio: 分层下采样比例(0~1)，默认1.0表示全量数据。
                       例如 0.2 表示取正负样本各 20%，保持类别比例不变。
    """
    N, D = feature_data.shape
    k = k_dim if k_dim else D

    Abnormaldata = feature_data[label_data != 0, :]
    Normaldata = feature_data[label_data == 0, :]

    np.random.shuffle(Normaldata)
    np.random.shuffle(Abnormaldata)

    # 分层下采样：从正负样本各自取 subsample_ratio 比例，保持分布均匀
    if subsample_ratio < 1.0:
        n_normal = int(len(Normaldata) * subsample_ratio)
        n_abnormal = int(len(Abnormaldata) * subsample_ratio)
        print(f"[OPT-DATA] 分层下采样: 正样本 {len(Normaldata)}→{n_normal}, 负样本 {len(Abnormaldata)}→{n_abnormal}")
        Normaldata = Normaldata[:n_normal]
        Abnormaldata = Abnormaldata[:n_abnormal]

    vacc_NonSelf = np.array_split(Abnormaldata, num_vaccines)
    vacc_Self = np.array_split(Normaldata, num_vaccines)

    #  初始化检测器（自动使用 GPU）
    detector = NetVaxDetector(k_dim=k, budget=100)
    if use_mahalanobis:
        all_data = np.vstack([Normaldata[:1000], Abnormaldata[:1000]])
        detector.fit_metric(all_data)

    metric_ref = detector.metric if use_mahalanobis else None

    # 第一批
    NonSelfs = vacc_NonSelf[0]
    Selfs = vacc_Self[0]
    Radius = Calculate_Radius(step, Selfs, NonSelfs,
                              use_mahalanobis=use_mahalanobis,
                              metric=metric_ref)
    mature_detector = np.c_[Selfs, Radius]

    detector.add_self_vaccine(Selfs)
    detector.add_nonself_vaccine(NonSelfs)

    if use_controlled_forgetting:
        for i in range(len(mature_detector)):
            det_id = detector.forgetting.add_detector(
                mature_detector[i, :-1], mature_detector[i, -1]
            )

    metrics_history = {'acc': [], 'prec': [], 'recall': [], 'f1score': []}

    test_nonself = vacc_NonSelf[-1]
    test_self = vacc_Self[-1]
    predict = detect(step, mature_detector[:, -1], mature_detector[:, :-1],
                     np.r_[test_nonself, test_self],
                     use_mahalanobis=use_mahalanobis, metric=metric_ref)
    true_labels = np.r_[np.ones(len(test_nonself)), np.zeros(len(test_self))]
    acc, p, r, f1 = getAccuracy(predict, true_labels)
    metrics_history['acc'].append(acc)
    metrics_history['prec'].append(p)
    metrics_history['recall'].append(r)
    metrics_history['f1score'].append(f1)

    for i in range(1, num_vaccines):
        print(f"\n=== 注入第 {i + 1}/{num_vaccines} 个疫苗 ===")

        mature_detector = tune_mature(step, mature_detector, vacc_NonSelf[i],
                                      use_mahalanobis=use_mahalanobis,
                                      metric=metric_ref)

        if use_controlled_forgetting:
            for idx, det_id in enumerate(detector.forgetting.detectors):
                if idx < len(mature_detector):
                    detector.forgetting.detectors[det_id]['radius'] = mature_detector[idx, -1]

        NonSelfs = np.r_[NonSelfs, vacc_NonSelf[i]]
        new_Radius = Calculate_Radius(step, vacc_Self[i], NonSelfs,
                                      use_mahalanobis=use_mahalanobis,
                                      metric=metric_ref)
        new_mature_detector = np.c_[vacc_Self[i], new_Radius]
        mature_detector = np.r_[mature_detector, new_mature_detector]

        detector.add_self_vaccine(vacc_Self[i])
        detector.add_nonself_vaccine(vacc_NonSelf[i])

        if use_controlled_forgetting:
            for j in range(len(new_mature_detector)):
                det_id, msg = detector.instantiate_detector(
                    new_mature_detector[j, :-1],
                    new_mature_detector[j, -1]
                )

        if use_controlled_forgetting:
            for det_id in list(detector.forgetting.detectors.keys()):
                detector.forgetting.update_memory_strength(det_id)
            pruned = detector.forgetting.prune()
            if pruned > 0:
                print(f"  [受控遗忘] 剪枝了 {pruned} 个检测器")
            alive_pairs = detector.forgetting.get_alive_detectors()
            if len(alive_pairs) > 0:
                mature_detector = np.array([
                    np.r_[c, r] for c, r in alive_pairs
                ])

        predict = detect(step, mature_detector[:, -1], mature_detector[:, :-1],
                         np.r_[test_nonself, test_self],
                         use_mahalanobis=use_mahalanobis, metric=metric_ref)
        acc, p, r, f1 = getAccuracy(predict, true_labels)
        metrics_history['acc'].append(acc)
        metrics_history['prec'].append(p)
        metrics_history['recall'].append(r)
        metrics_history['f1score'].append(f1)

        print(f"  当前检测器数量: {len(mature_detector)}")
        #  每个疫苗注入周期后完整清理 GPU 显存
        clear_gpu_memory()

    return detector, metrics_history


from LoadData import *

# %% ── 入口 ──
if __name__ == "__main__":
    print("=" * 60)
    print("优化版 NetVax — 向量化 + GPU 加速")
    print("=" * 60)

    feature_data, label_data = load_NF_dataset_cached(
        'NF-ToN-IoT','./NF-ToN-IoT-v3/data/NF-ToN-IoT-v3.csv')
    print(f"\n数据形状: features={feature_data.shape}, labels={label_data.shape}")
    print(f"正样本(自体): {np.sum(label_data == 0)}, 负样本(非自体): {np.sum(label_data != 0)}")

    detector, metrics = run_netvax_pipeline(
        feature_data, label_data,
        num_vaccines=5, step=5000,
        use_mahalanobis=True,
        use_controlled_forgetting=True,
        subsample_ratio=0.1
    )

    print("\n实验结果:")
    for key, vals in metrics.items():
        print(f"  {key}: {[f'{v:.2f}' for v in vals]}")
