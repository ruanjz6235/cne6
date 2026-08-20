# backtest_py.py
import numpy as np
import cvxpy as cp

def rolling_optimization(returns, lookback=252, gamma=1.0):
    """
    returns: (T_total, n) 收益率矩阵
    lookback: 估计窗口长度
    gamma: 风险厌恶系数
    返回: portfolio_returns (T_total - lookback,) 组合日收益序列
    """
    T, n = returns.shape
    port_ret = np.empty(T - lookback)
    prev_w = np.ones(n) / n   # 初始等权，仅用于记录

    for t in range(lookback, T):
        # 1. 用过去 lookback 天的数据估计
        hist = returns[t-lookback:t]
        mu = np.mean(hist, axis=0)
        Sigma = np.cov(hist, rowvar=False)

        # 2. 构建并求解凸优化问题
        w = cp.Variable(n, nonneg=True)
        objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))
        constraints = [cp.sum(w) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        # 3. 获取最优权重（若求解失败则沿用上一期权重）
        if w.value is None:
            w_opt = prev_w
        else:
            w_opt = w.value

        # 4. 计算本期（t时刻）的组合收益
        port_ret[t - lookback] = np.dot(w_opt, returns[t])
        prev_w = w_opt

    return port_ret

if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    T = 1000
    n = 8
    returns = np.random.randn(T, n) * 0.01 + 0.0002

    import time
    time1 = time.time()
    port_rets = rolling_optimization(returns, lookback=252, gamma=2.0)
    print(time.time() - time1)
    cum_ret = np.cumprod(1 + port_rets) - 1
    print(f"累计收益: {cum_ret[-1]:.4f}")
