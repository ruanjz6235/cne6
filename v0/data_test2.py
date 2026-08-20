import numpy as np
import pandas as pd
from arch import arch_model
import warnings

def predict_factor_volatility_ar_garch(
    factor_returns: pd.DataFrame,
    horizon: int = 21,
    min_periods: int = 252,
    fallback_decay: float = 0.06
) -> pd.Series:
    """
    使用 AR(1)-GARCH(1,1) 模型预测因子未来 horizon 期的波动率（标准差）。

    Parameters
    ----------
    factor_returns : pd.DataFrame
        因子日收益率，形状 (T, N)，index 为日期，columns 为因子名称。
    horizon : int
        预测期数，默认 21（约一个月）。
    min_periods : int
        最少需要的样本数量，过短则回退到简单波动率方法。
    fallback_decay : float
        当 GARCH 不收敛时，EWMA 的衰减因子（对应于半衰期约为 log(0.5)/log(1-decay) ≈ 11 天）。

    Returns
    -------
    predicted_vol : pd.Series
        各因子的预测波动率，index 为因子名称。
    """
    if factor_returns.empty or factor_returns.shape[0] < min_periods:
        raise ValueError(f"因子收益率数据不足，至少需要 {min_periods} 期。")

    # 去除全为 NaN 的列，并用前向填充处理个别缺失值
    factor_returns = factor_returns.dropna(axis=1, how='all').ffill().bfill()
    factor_names = factor_returns.columns
    N = len(factor_names)
    returns = factor_returns.values  # (T, N)

    # ============================
    # 1. 向量化估计 AR(1) 参数与残差
    # ============================
    y = returns[1:, :]      # (T-1, N)  当期
    x = returns[:-1, :]     # (T-1, N)  滞后一期

    # 每个因子的均值
    mean_y = y.mean(axis=0)
    mean_x = x.mean(axis=0)

    # 方差与协方差 (有偏估计，与 OLS 一致)
    var_x = np.mean((x - mean_x) ** 2, axis=0)
    cov_xy = np.mean((y - mean_y) * (x - mean_x), axis=0)

    # AR(1) 系数
    # 对于方差接近零的因子（常数序列），令 phi = 0, c = mean_y
    valid_var = var_x > 1e-12
    phi = np.zeros(N)
    phi[valid_var] = cov_xy[valid_var] / var_x[valid_var]
    c = mean_y - phi * mean_x

    # 残差序列 (T-1, N)
    epsilon = y - (c + phi * x)

    # ============================
    # 2. 逐因子拟合 GARCH(1,1)
    # ============================
    omega = np.zeros(N)
    alpha = np.zeros(N)
    beta  = np.zeros(N)
    sigma2_last = np.zeros(N)
    epsilon_last = np.zeros(N)
    converged = np.full(N, False)   # 标记是否成功拟合

    # 禁用 arch 库在拟合过程中的打印信息
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    for i in range(N):
        eps_series = epsilon[:, i]
        # 若残差几乎无波动，直接使用样本标准差作为预测
        if np.std(eps_series) < 1e-12:
            sigma2_last[i] = np.var(eps_series)
            epsilon_last[i] = eps_series[-1]
            omega[i] = sigma2_last[i]
            alpha[i] = 0.0
            beta[i] = 0.0
            converged[i] = True
            continue

        try:
            # 拟合 GARCH(1,1)，均值方程设为常数零（因残差已去均值）
            model = arch_model(eps_series, mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
            # 使用较稳健的优化器，适当增加最大迭代次数
            res = model.fit(disp='off', options={'maxiter': 500}, show_warning=False)

            # 提取参数：omega, alpha[1], beta[1]
            params = res.params
            omega[i] = params['omega']
            alpha[i] = params['alpha[1]']
            beta[i]  = params['beta[1]']

            # 最后一日残差
            epsilon_last[i] = eps_series[-1]
            # 最后一日条件方差（var(epsilon_t | F_{t-1})）
            sigma2_last[i] = res.conditional_volatility[-1] ** 2
            converged[i] = True

        except Exception:
            # 若 GARCH 不收敛，回退到 EWMA 方差预测
            converged[i] = False

    # 对不收敛的因子使用 EWMA 作为备用方案
    if not np.all(converged):
        not_conv = ~converged
        # 用全样本残差计算无条件方差作为初始值
        sample_var = np.var(epsilon[:, not_conv], axis=0)
        # 用 EWMA 递推得到最后一日的条件方差
        ewma_sigma2 = np.copy(sample_var)
        ewma_decay = fallback_decay
        for t in range(epsilon.shape[0]):
            sq = epsilon[t, not_conv] ** 2
            ewma_sigma2 = ewma_decay * sq + (1 - ewma_decay) * ewma_sigma2
        sigma2_last[not_conv] = ewma_sigma2
        epsilon_last[not_conv] = epsilon[-1, not_conv]
        # 设置 omega 为样本方差 * (1 - alpha - beta) 的一个近似，这里直接令 omega = 样本方差
        omega[not_conv] = sample_var
        alpha[not_conv] = 0.0
        beta[not_conv] = 0.0

    # ====================================================
    # 3. 向量化多期预测：计算未来 horizon 期的总方差
    # ====================================================
    # 预分配状态变量 (N,)
    V = np.zeros(N)          # var(r_{T+k} | F_T)
    C = np.zeros(N)          # Cov(S_{k-1}, r_{T+k} | F_T)
    W = np.zeros(N)          # Var(S_k | F_T)，S_k = sum_{j=1}^k r_{T+j}

    # 第一步的条件方差预测：sigma2_{T+1|T} = omega + alpha * epsilon_T^2 + beta * sigma2_T
    sigma2_pred = omega + alpha * (epsilon_last ** 2) + beta * sigma2_last
    # 防止数值问题导致负数
    sigma2_pred = np.maximum(sigma2_pred, 1e-12)

    # 递推 horizon 步
    for k in range(1, horizon + 1):
        # 本期条件方差 var(r_{T+k})
        V = phi * phi * V + sigma2_pred
        # 累积收益方差
        W = W + V + 2.0 * C
        # 更新协方差 C_{k+1}
        C = phi * (C + V)
        # 更新下一期的预测方差：sigma2_{T+k+1|T} = omega + (alpha+beta) * sigma2_{T+k|T}
        sigma2_pred = omega + (alpha + beta) * sigma2_pred
        sigma2_pred = np.maximum(sigma2_pred, 1e-12)

    # 预测波动率 = sqrt(总方差)
    predicted_vol = np.sqrt(np.maximum(W, 0))

    return pd.Series(predicted_vol, index=factor_names, name='predicted_volatility')


import numpy as np
import pandas as pd
from numpy.testing import assert_array_less, assert_equal

# 假设上面的函数已经定义或从模块导入
# from your_module import predict_factor_volatility_ar_garch

def generate_synthetic_factor_returns(
    n_days: int = 500,
    n_factors: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成含有轻度自相关和波动率聚类的合成因子收益率，以便测试 AR-GARCH 预测流程。
    """
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
    factor_names = [f'Factor_{i}' for i in range(1, n_factors + 1)]

    # 用随机参数生成 AR(1)-GARCH(1,1) 过程
    phi_true = np.array([0.1, -0.05, 0.0, 0.2, -0.15])
    omega_true = np.array([0.05, 0.08, 0.1, 0.03, 0.06])
    alpha_true = np.array([0.1, 0.15, 0.08, 0.12, 0.1])
    beta_true = np.array([0.8, 0.7, 0.85, 0.8, 0.75])

    returns = np.zeros((n_days, n_factors))
    sigma2 = np.zeros((n_days, n_factors))
    eps = np.zeros((n_days, n_factors))

    # 初始方差设为无条件方差
    sigma2[0, :] = omega_true / (1 - alpha_true - beta_true)
    eps[0, :] = np.sqrt(sigma2[0, :]) * np.random.randn(n_factors)
    returns[0, :] = eps[0, :]  # 假设初始均值为0

    for t in range(1, n_days):
        # 条件均值：AR(1)
        mu = phi_true * returns[t - 1, :]
        # 条件方差：GARCH(1,1)
        sigma2[t, :] = omega_true + alpha_true * (eps[t - 1, :] ** 2) + beta_true * sigma2[t - 1, :]
        eps[t, :] = np.sqrt(sigma2[t, :]) * np.random.randn(n_factors)
        returns[t, :] = mu + eps[t, :]

    return pd.DataFrame(returns, index=dates, columns=factor_names)


def test_predict_factor_volatility_ar_garch():
    """
    测试多因子波动率预测函数。
    验证：
    1. 返回类型为 pd.Series，索引与输入因子名一致。
    2. 所有预测波动率为非负数。
    3. 预测值处于合理范围（不会极大或为零）。
    4. 能正确处理最小样本数量不足的情况（可选）。
    """
    # ---------- 生成模拟数据 ----------
    n_days = 500
    n_factors = 5
    factor_returns = generate_synthetic_factor_returns(n_days, n_factors)

    # ---------- 调用预测函数 ----------
    horizon = 21
    predicted_vol = predict_factor_volatility_ar_garch(
        factor_returns,
        horizon=horizon,
        min_periods=252
    )

    # ---------- 断言检查 ----------
    # 1. 类型与形状
    assert isinstance(predicted_vol, pd.Series), "输出应为 pd.Series"
    assert predicted_vol.shape[0] == n_factors, "输出长度应与因子数一致"
    assert_equal(predicted_vol.index.tolist(), factor_returns.columns.tolist())

    # 2. 所有预测波动率非负
    assert (predicted_vol >= 0).all(), "预测波动率不应为负"

    # 3. 波动率合理范围（年化通常在 5%~80% 之间，此处只做极宽泛检查）
    daily_vol = predicted_vol.values  # horizon=21 天的预测波动率，应为 21 日标准差
    # 转换为年化大致判断：允许在 1%~200% 之间
    annual_vol = daily_vol * np.sqrt(252 / horizon)
    assert (annual_vol > 0.01).all(), "年化波动率应 > 1%"
    assert (annual_vol < 2.0).all(), "年化波动率应 < 200%"

    print("所有测试通过！")
    print(predicted_vol)


if __name__ == "__main__":
    test_predict_factor_volatility_ar_garch()

# 假设 factor_returns 是一个 pd.DataFrame，行为日期，列为 CNE6 因子名称
# predicted_vol = predict_factor_volatility_ar_garch(factor_returns, horizon=21)


