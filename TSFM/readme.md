Conformal Prediction（CP）：概述

Conformal Prediction（CP）是一种用于预测和不确定性量化的数学方法。它具有模型无关（model-agnostic）和分布无关（distribution-free）的特性，意味着可以与多种机器学习模型（包括神经网络等预训练模型）结合使用，为预测提供置信度和不确定性量化信息。

https://github.com/ibm-granite/granite-tsfm/blob/main/tsfm_public/toolkit/conformal.py
这个代码的核心目标是采用后验概率方法 (Post-Hoc Probabilistic approaches)，将时间序列预测模型的点估计 (point estimates) 转化为概率预测 (probabilistic forecasts)，特别是计算目标变量的分位数估计 (quantile estimate)

您好，根据您提供的代码片段，这是一个关于**保形预测 (Conformal Forecasting)** 及其相关实用工具和处理器的 Python 代码库的解释。

该代码库的核心目标是采用**后验概率方法 (Post-Hoc Probabilistic approaches)**，将时间序列预测模型的**点估计 (point estimates)** 转化为**概率预测 (probabilistic forecasts)**，特别是计算目标变量的**分位数估计 (quantile estimate)**。

以下是对主要组件和功能的详细解释：（gemini generated）

---

### 1. 核心处理器：`PostHocProbabilisticProcessor`

这是整个保形预测流程的入口点。它通过继承 `BaseProcessor` 并采用 Hugging Face (HF) 的 `FeatureExtractionMixin` 来支持序列化和反序列化。

#### 关键初始化参数 (`__init__`):

*   **`quantiles` (分位数):** 目标分位数的列表，例如 `[0.1, 0.9]`，用于计算预测区间。
*   **`method` (方法):** 决定使用的后验概率方法。主要有两种：
    *   **`CONFORMAL` (保形):** 使用保形预测方法，由 `WeightedConformalForecasterWrapper` 实现。
    *   **`GAUSSIAN` (高斯):** 假设残差服从独立高斯分布，由 `PostHocGaussian` 实现。
*   **`nonconformity_score` (非一致性分数):** 仅在选择保形方法时适用。定义了如何衡量预测误差。可选值包括：
    *   `ABSOLUTE_ERROR` (绝对误差)。
    *   `ERROR` (有符号误差 $y - y_{pred}$)。
*   **`weighting` (加权策略):** 用于加权非一致性分数。可选策略包括 `UNIFORM` (均匀加权) 和 `EXPONENTIAL_DECAY` (指数衰减)。
*   **`window_size` (窗口大小):** 考虑过去残差的最大上下文窗口大小。如果为 `None`，则使用所有过去的观测值。
*   **`aggregation` 和 `aggregation_axis` (聚合):** 定义了如何跨维度（如预测步长或特征数）聚合异常值分数。

#### 核心流程方法:

*   **`train` (训练/拟合):** 使用校准数据（`y_cal_gt` 为真实值，`y_cal_pred` 为模型预测值）拟合后验概率包装器。如果输入是多时间序列，它会根据 `id_columns` 为每个唯一的 ID 组初始化并拟合一个模型。
    *   对于保形方法，它会检查所需的最小校准点数 (`self.critical_size`)，以确保校准集足够大。
*   **`predict` (预测):** 接收测试预测值 (`y_test_pred`)，输出概率预测结果 (`y_test_prob_pred`)，其维度包含分位数。
*   **`update` (更新):** 允许模型使用新的真实值 (`y_gt`) 和预测值 (`y_pred`) 在线更新其内部状态（例如，更新保形方法中的非一致性分数）。
*   **`outlier_score` (异常值分数):** 计算基于预测误差的**标准化异常值保形分数 (P-value)**。

---

### 2. 保形预测方法：`WeightedConformalForecasterWrapper` 和 `WeightedConformalWrapper`

`WeightedConformalForecasterWrapper` 是多变量预测的包装器，它为每个**预测步长**和每个**特征**初始化一个**单变量保形包装器** (`WeightedConformalWrapper`)。

#### 2.1 核心概念：非一致性分数和集合

*   **`nonconformity_score_functions`:** 用于计算非一致性分数 $\alpha_i$。例如，如果使用 `ABSOLUTE_ERROR`，分数是 $|y_{gt} - y_{pred}|$。
*   **`conformal_set`:** 使用计算出的**分数阈值** (`score_threshold`) 和点预测 (`y_pred`) 来确定保形预测集或区间。
    *   例如，对于绝对误差，预测区间为 $[y_{pred} - \text{threshold}, y_{pred} + \text{threshold}]$。

#### 2.2 阈值计算 (`WeightedConformalWrapper`):

`WeightedConformalWrapper` 存储校准分数 (`self.cal_scores`) 和权重 (`self.weights`)。

*   **`get_weights`:** 根据选定的加权策略（如均匀或指数衰减）返回校准权重。
*   **`score_threshold_func`:** 计算与所需**假警报率** (`false_alarm` 或 $\alpha$) 对应的非一致性分数阈值。
    *   该函数内部调用 **`weighted_conformal_quantile`**。
    *   `weighted_conformal_quantile` 通过对排序后的分数和对应的权重进行累积求和，找到使得累积权重超过 $1 - \alpha$ 的最小分数，作为保形分位数 (阈值)。

#### 2.3 在线优化 (Adaptive Weighting):

如果启用了权重优化 (如 `WASS1` - Wasserstein-1 距离优化)，则使用 `AdaptiveWeightedConformalScoreWrapper`。

*   该自适应包装器利用 PyTorch (`torch`) 进行权重优化，旨在最小化经验分布（由加权保形 P-value 得到）与均匀分布之间的 Wasserstein-1 距离。
*   `get_beta` 函数计算加权保形 p-分数 (beta_t)。

---

### 3. 高斯方法：`PostHocGaussian`

当 `method` 为 `GAUSSIAN` 时使用此包装器。它将点预测转换为概率预测，但前提是假设残差服从独立的高斯分布。

*   **拟合 (`fit`):** 计算校准误差 (`self.errors`)，并基于这些误差估计**方差** (`self.variance`)。
*   **预测 (`predict`):**
    1.  计算**标准差** (`std_devs`)。
    2.  使用标准正态分布的**百分点函数** (`norm.ppf`) 找到每个目标分位数的 Z 分数。
    3.  将 Z 分数乘以标准差，然后加到点预测值 (`y_test_pred`) 上，从而得到分位数估计。

---

### 总结要点

该代码库的核心在于提供了一个灵活的框架，用于对时间序列预测结果进行后验概率校准。它支持两种主要的后验方法（保形和高斯），并且在保形方法中提供了细致的控制，包括多种非一致性分数类型 (`absolute_error`, `error`) 和加权策略 (`uniform`, `exponential_decay`)，甚至支持基于 Wasserstein-1 距离的权重自适应优化。这使得预测者能够为给定的点预测生成可靠的**预测区间**或**分位数估计**。




应用的时候对数据类型有硬性要求：
1. 数据类型要求
校准数据的输入类型必须是以下两者之一：
NumPy 数组 (np.ndarray)。
Pandas DataFrame (pd.DataFrame)。
硬性要求：
• 如果 y_cal_gt (真实值) 是 pd.DataFrame，那么 y_cal_pred (预测值) 也必须是 pd.DataFrame。
• 如果其中一个不是 pd.DataFrame，那么它们都应该被视为 np.ndarray。
2. 数据维度和形状要求
无论输入是 DataFrame 还是 np.ndarray，在内部处理时，数据最终都会被转换为 NumPy 数组，并且必须满足严格的 3 维形状要求。
硬性要求（NumPy 数组的形状）：
• y_cal_gt (真实值) 和 y_cal_pred (预测值) 都必须是 3 维数组。
• 期望的形状为：  \text{nsamples} \times \text{forecast_horizon} \times \text{number_features}  （即：样本数 × 预测步长 × 特征数/变量数）

多时间序列处理
如果处理的是多时间序列数据（通过设置 id_columns），那么输入数据的结构和内容有特殊要求：
• 如果输入是 pd.DataFrame： DataFrame 必须包含用于识别不同时间序列的 ID 列。
    ◦ 在训练时，y_cal_pred（以及 y_cal_gt）会使用这些 id_columns 进行分组，并为每个唯一的 ID 组拟合一个独立的模型。
• 如果输入是 np.ndarray： 必须提供一个额外的参数 id_column_values (包含 ID 列值的 NumPy 数组)，用于识别不同的时间序列
