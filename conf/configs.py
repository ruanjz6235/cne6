"""
Barra CNE6 配置类
==================
包含所有配置数据类
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RawDataConfig:
    """数据配置"""
    start_date: str
    end_date: str
    market: str = 'A'


@dataclass
class PreprocessConfig:
    """预处理配置"""
    winsorize_method: str = 'mad'       # mad/percentile/sigma
    winsorize_n: float = 3.0            # 去极值参数
    neutralize_method: str = 'industry' # industry/market/both
    standardize_method: str = 'zscore'  # zscore/rank


@dataclass
class FactorConfig:
    """因子配置"""
    window: int = 504           # Beta 计算窗口
    beta_half_life: int = 252         # 权重半衰期
    momentum_window: int = 252       # 动量计算窗口
    volatility_window: int = 252     # 波动率计算窗口

    @staticmethod
    def decay_lambda(half_life: int = None) -> float:
        """指数衰减因子 λ = ln(0.5) / half_life"""
        return np.log(0.5) / half_life

    def get_weights(self,
                    n_periods: int = None,
                    half_life: int = None
                    ) -> np.ndarray:
        """
        生成指数衰减权重向量

        权重公式：w(t) = exp(λ * (T - 1 - t))
        - t=0 对应最早的数据点
        - t=T-1 对应最近的数据点
        - 最近的数据权重更大（半衰期衰减）

        Parameters
        ----------
        n_periods : int, optional
            期间数，默认为 self.window

        Returns
        -------
        np.ndarray
            权重向量
        """
        if n_periods is None:
            n_periods = self.window
        if half_life is None:
            half_life = self.beta_half_life
        t = np.arange(n_periods)
        weights = np.exp(self.decay_lambda(half_life=half_life) * (n_periods - 1 - t))
        return weights

    def normalized_weights(self,
                           n_periods: int = None,
                           half_life: int = None
                           ) -> np.ndarray:
        """获取归一化权重（Σw = 1）"""
        w = self.get_weights(n_periods, half_life)
        return w / np.sum(w)

if __name__ == '__main__':
    config = FactorConfig()
    config.normalized_weights(n_periods=21, half_life=5)



