"""
Barra CNE6 结果类
==================
包含所有返回数据类
"""

from dataclasses import dataclass
import pandas as pd


@dataclass
class BetaResult:
    """Beta 回归结果"""
    beta: pd.DataFrame          # Beta 系数 (n_periods × n_stocks)
    alpha: pd.DataFrame         # Alpha 截距
    r_squared: pd.DataFrame     # R 平方拟合优度
    residual_std: pd.DataFrame  # 残差标准差
    beta_se: pd.DataFrame       # Beta 标准误
    t_stat: pd.DataFrame        # Beta t 统计量

    def concat(self, result, axis=1):
        self.beta = pd.concat([self.beta, result.beta], axis=axis)
        self.alpha = pd.concat([self.alpha, result.alpha], axis=axis)
        self.r_squared = pd.concat([self.r_squared, result.r_squared], axis=axis)
        self.residual_std = pd.concat([self.residual_std, result.residual_std], axis=axis)
        self.beta_se = pd.concat([self.beta_se, result.beta_se], axis=axis)
        self.t_stat = pd.concat([self.t_stat, result.t_stat], axis=axis)
