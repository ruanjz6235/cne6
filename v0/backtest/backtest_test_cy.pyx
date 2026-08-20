# backtest_cy.pyx
import numpy as np
cimport numpy as np
cimport cython
from cvxpy import Variable, Problem, Maximize, quad_form, sum as cp_sum, ECOS

@cython.boundscheck(False)
@cython.wraparound(False)
def rolling_optimization_cy(double[:, ::1] returns,
                            int lookback=252,
                            double gamma=1.0):
    """
    Cython 加速的多期滚动优化回测。
    returns: memoryview of shape (T, n), C-order.
    返回: tuple (portfolio_returns, final_weights)
    """
    cdef int T = returns.shape[0]
    cdef int n = returns.shape[1]
    cdef int t, i, j
    cdef double[:] port_ret = np.empty(T - lookback)
    cdef double[:] mu = np.empty(n)
    cdef double[:, ::1] Sigma = np.empty((n, n))
    cdef double[:] prev_w = np.ones(n) / n
    cdef double[:] w_opt

    # 预先分配用于存储历史窗口的临时数组
    cdef double[:, ::1] hist = np.empty((lookback, n))

    for t in range(lookback, T):
        # 拷贝历史窗口数据（避免 Python 切片开销）
        for i in range(lookback):
            for j in range(n):
                hist[i, j] = returns[t - lookback + i, j]

        # 计算均值
        for j in range(n):
            mu[j] = 0.0
            for i in range(lookback):
                mu[j] += hist[i, j]
            mu[j] /= lookback

        # 计算协方差
        for j in range(n):
            for i in range(n):
                Sigma[j, i] = 0.0
                for k in range(lookback):
                    Sigma[j, i] += (hist[k, j] - mu[j]) * (hist[k, i] - mu[i])
                Sigma[j, i] /= (lookback - 1)

        # 调用 CVXPY 构建并求解（这是 Python 调用，但循环其他部分已加速）
        w = Variable(n, nonneg=True)
        objective = Maximize(
            np.asarray(mu) @ w - gamma * quad_form(w, np.asarray(Sigma))
        )
        constraints = [cp_sum(w) == 1]
        prob = Problem(objective, constraints)
        prob.solve(solver=ECOS, verbose=False)

        if w.value is None:
            w_opt = np.asarray(prev_w)
        else:
            w_opt = w.value

        # 计算本日组合收益
        port_ret[t - lookback] = 0.0
        for j in range(n):
            port_ret[t - lookback] += w_opt[j] * returns[t, j]

        # 更新上一期权重
        for j in range(n):
            prev_w[j] = w_opt[j]

    return np.asarray(port_ret)