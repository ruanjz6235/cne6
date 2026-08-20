import numpy as np
from backtest_test_cy import rolling_optimization_cy

np.random.seed(42)
T = 1000
n = 8
returns = np.random.randn(T, n) * 0.01 + 0.0002

import time
time1 = time.time()
port_rets = rolling_optimization_cy(returns, lookback=252, gamma=2.0)
print(time.time() - time1)
cum_ret = np.cumprod(1 + port_rets) - 1
print(f"累计收益: {cum_ret[-1]:.4f}")
