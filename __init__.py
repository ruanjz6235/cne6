"""
Barra CNE6 因子计算模块
========================

Barra 中国股权市场 6 版因子计算系统

模块结构:
-----------
1. configs.py              - 配置类 (RawDataConfig, PreprocessConfig, FactorConfig)
2. barra_raw_data.py       - 取数类 (BarraRawDataLoader)
3. barra_base_functions.py - 基础函数类 (BarraBaseFunctions, BarraUtils)
4. barra_factors.py        - 因子类 (9 大风格因子 + BarraFactorFactory)
5. barra_calculator.py     - 计算器类 (BarraCNE6Calculator)

使用示例:
-----------
```python
from Barra.cne import BarraCNE6Calculator, PreprocessConfig, FactorConfig

# 配置
cfg = PreprocessConfig(winsorize_method='mad', winsorize_n=3.0)
factor_cfg = FactorConfig(beta_window=252, beta_half_life=63)

# 初始化
calc = BarraCNE6Calculator('20250101', '20251231', cfg, factor_cfg)

# 加载数据
calc.load_data_from_file('data.feather')

# 计算因子
calc.calculate_factors()

# 获取结果
factors = calc.get_factor_data()

# 保存
calc.save_results('output.pkl')
```

因子列表:
-----------
- Size (尺寸因子)
- Beta (贝塔因子)
- Momentum (动量因子)
- Value (价值因子)
- Volatility (波动率因子)
- Profitability (盈利能力因子)
- Growth (成长因子)
- Liquidity (流动性因子)
- Leverage (杠杆因子)
"""

from app.model.v5.cne6.conf.configs import RawDataConfig, PreprocessConfig, FactorConfig
from app.model.v5.cne6.barra_raw_data import BarraRawData
from app.model.v5.cne6.barra_base_functions import BarraBaseFunctions
from app.model.v5.cne6.barra_factors import (
    BarraFactorFactory,
    SizeFactor,
    MomentumFactor,
    ValueFactor,
    VolatilityFactor,
    GrowthFactor,
    LiquidityFactor,
)
from app.model.v5.cne6.barra_calculator import BarraCNE6Calculator

__version__ = '1.0.0'
__all__ = [
    # Configs
    'RawDataConfig',
    'PreprocessConfig',
    'FactorConfig',
    # Raw Data
    'BarraRawData',
    # Base Functions
    'BarraBaseFunctions',
    # Factors
    'BarraFactorFactory',
    'SizeFactor',
    'MomentumFactor',
    'ValueFactor',
    'VolatilityFactor',
    'GrowthFactor',
    'LiquidityFactor',
    # Calculator
    'BarraCNE6Calculator',
]

print(f"Barra CNE6 模块加载成功！版本：{__version__}")
