# Barra CNE6 多因子风险模型

基于 Barra CNE6 模型的多因子风险模型实现，包含因子暴露计算和因子收益率计算两大核心模块。

## 项目结构

```
cne6/
├── configs.py              # 配置类（已弃用，迁移到enum目录）
├── barra_raw_data.py       # 数据取数模块（数据库/文件加载）
├── barra_base_functions.py # 基础函数类（数据预处理、加权回归）
├── barra_factors.py        # 因子计算类（8大风格因子）
├── barra_calculator.py     # 因子暴露计算器（主程序入口）
├── barra_factor_return_calculator.py # 因子收益率计算器
├── __init__.py             # 模块初始化文件
├── enum/                   # 配置与结果类目录
│   ├── configs.py          # 配置类（数据配置、预处理配置、因子配置）
│   └── result.py           # 结果数据类（BetaResult）
├── example/                # 测试示例目录
│   ├── barra_test.py       # 测试数据生成模块
│   ├── barra_model_test.py # 模型测试模块
│   ├── barra_test_data.pkl # 测试行情数据文件
│   └── barra_industry_test_data.pkl # 测试行业数据文件
├── data/                   # 数据目录
│   └── barra中台供数.xlsx   # 数据需求说明文件
├── doc/                    # 文档目录
│   ├── CNLT&CNTR因子介绍 V2.pdf
│   ├── Comparison Datasheet - Barra China A Total Market Equity Model.pdf
│   └── Descriptor Details - Barra China A Total Market Equity Model.pdf
└── README.md               # 项目说明文档
```

## 使用教程

### 步骤一：生成测试数据（pkl文件）

运行 `barra_test.py` 生成模拟数据：

```bash
python barra_test.py
```

按提示输入 `y` 保存数据，生成以下文件：
- `barra_test_data.pkl` - 股票行情与财务数据（约 2000 股票 × 1500 交易日）
- `barra_industry_test_data.pkl` - 行业分类数据

**数据字段说明：**

| 字段 | 说明 |
|------|------|
| secu_code | 股票代码 |
| end_date | 交易日期 |
| market_value | 总市值 |
| pct_change | 个股日收益率 |
| close | 收盘价 |
| index_return | 指数收益率 |
| pred_divd_rate | 预测股息率 |
| divd_rate_ttm | 股息率TTM |
| bp | 账面市值比(1/PB) |
| net_profit_ttm | 净利润TTM |
| ebit_ttm | EBIT TTM |
| corp_val_crcp | 企业价值 |
| total_liability | 总负债 |
| total_asset | 总资产 |
| total_income_ttm | 营业收入TTM |
| total_cost_ttm | 营业成本TTM |
| cash_flow_ttm | 经营性现金流TTM |
| circulate_shares | 流通股本 |
| capex | 资本支出 |
| mth_tnrt | 月换手率 |
| quar_tnrt | 季换手率 |
| ann_tnrt | 年换手率 |

### 步骤二：计算因子暴露和因子收益率

运行 `barra_model.py` 执行完整模型测试：

```bash
python barra_model.py
```

测试流程：
1. 加载 pkl 数据文件
2. 计算 8 大风格因子暴露
3. 进行因子预处理（去极值、标准化）
4. 计算因子收益率（加权最小二乘回归）

---

## 模型架构

### 整体流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Barra CNE6 模型流程                            │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌───────────────┐    ┌────────────┐    ┌───────────┐
  │  数据取数  │ -> │  基础函数类    │ -> │  因子计算   │ -> │ 因子收益率 │
  │          │    │  (预处理)     │    │            │    │   计算    │
  └──────────┘    └───────────────┘    └────────────┘    └───────────┘

  1. 取数阶段:
     - 从数据库/文件加载原始行情、财务、预测数据
     - 加载行业分类数据（申万2021）
     - 字段标准化、日期筛选

  2. 预处理阶段:
     - 数据清洗（缺失值填充、去重）
     - 去极值（MAD/百分位数/标准差方法）
     - 中性化（行业/市值回归取残差）
     - 标准化（Z-Score/Rank）

  3. 因子计算阶段:
     - 计算 8 大风格因子暴露
     - 每个因子由多个子因子合成

  4. 因子收益率阶段:
     - 构建因子暴露矩阵
     - 加权最小二乘回归
     - 行业约束条件
```

---

### 8大风格因子计算方法

#### 1. Size 尺寸因子

| 子因子 | 计算公式 | 说明 |
|--------|----------|------|
| LNCAP | ln(市值) | 对数市值 |
| MIDCAP | LNCAP³ - LNCAP | 中盘因子 |

**合成方式：** Size = mean(LNCAP, MIDCAP)

---

#### 2. Volatility 波动率因子

| 子因子 | 计算公式 | 说明 |
|--------|----------|------|
| Beta | 滚动加权回归 β | 个股相对指数的Beta |
| HSIGMA | 残差标准差 | 回归残差波动率 |
| DSTD | EWM(halflife=42).std() | 日收益率波动率 |
| CMRA | max(rs) - min(rs) | 累积收益率范围 |

**合成方式：** Volatility = mean(Beta, HSIGMA, DSTD, CMRA)

**Beta计算：** 使用半衰期加权最小二乘回归（WLS）
- 窗口期：504天
- 半衰期：252天
- 权重函数：w(t) = exp(λ × (T-1-t))

---

#### 3. Momentum 动量因子

| 子因子 | 计算公式 | 说明 |
|--------|----------|------|
| RSTR | EWM(halflife=126).mean() | 相对强度（超额收益滚动均值） |
| HALPHA | Beta回归截距 | 滚动回归Alpha |
| STREV | EWM(halflife=5).mean() | 短期反转 |

**合成方式：**
- sub_momentum = mean(RSTR, HALPHA)
- Momentum = mean(sub_momentum, STREV)

---

#### 4. Dividend Yield 红利因子

| 子因子 | 计算公式 | 说明 |
|--------|----------|------|
| DY | mean(pred_divd_rate, divd_rate_ttm) | 股息率 |

**合成方式：** Dividend Yield = DY

---

#### 5. Value 价值因子

| 子因子 | 计算公式 | 说明 |
|--------|----------|------|
| BP | 1/PB | 账面市值比 |
| EPTTM | 净利润TTM / 市值 | 盈利收益率 |
| EBIT_EV | EBIT TTM / 企业价值 | 息税前利润/企业价值 |
| pred_pera_avg | 分析师预测E/P | 预测市盈率倒数 |
| LTRSTR | 长期相对强度 | 长期动量 |
| LTHALPHA | 长期Alpha | 长期回归截距 |

**合成方式：** Value = mean(BP, EPTTM, EBIT_EV, pred_pera_avg, LTRSTR, LTHALPHA)

---

#### 6. Quality 质量因子（分层合成）

质量因子采用分层合成结构，包含5个子维度：

**第一层：子因子计算**

| 子维度 | 子因子 | 计算公式 |
|--------|--------|----------|
| Leverage | DTOA | 总负债 / 总资产 |
| Profitability | AT | 营业收入TTM / 总资产 |
| | GP | (营业收入-成本) / 总资产 |
| | ROA | 净利润TTM / 总资产 |
| Earnings Variability | VOS | 5年营收波动性 |
| | VOE | 5年利润波动性 |
| | VOC | 5年现金流波动性 |
| | PRED_VOEP | 预测EPS标准差 / 价格 |
| Earnings Quality | ABS | -应计项目 / 总资产 |
| Investment Quality | TAG | -5年资产增长斜率 |
| | IG | -5年流通股增长斜率 |
| | CEG | -5年资本支出增长斜率 |

**5年波动性计算：**
```
volatility = std(5年数据) / mean(5年数据)
```

**5年回归斜率计算：**
```
slope = WLS回归斜率 / mean(5年数据)
取负值以符合因子方向（低增长公司得分高）
```

**合成方式：** Quality = mean(所有子因子)

---

#### 7. Growth 成长因子

| 子因子 | 计算公式 | 说明 |
|--------|----------|------|
| EGRLF | 预测净利润增长率 | 分析师预测 |
| SGRO | 5年营收增长斜率 | 每股营收增长 |
| EGRO | 5年EPS增长斜率 | 每股收益增长 |

**合成方式：** Growth = mean(EGRLF, SGRO, EGRO)

---

#### 8. Liquidity 流动性因子

| 子因子 | 计算公式 | 说明 |
|--------|----------|------|
| STOM | ln(月换手率 + 1) | 月度流动性 |
| STOQ | ln(季换手率 + 1) | 季度流动性 |
| STOA | ln(年换手率 + 1) | 年度流动性 |

**合成方式：** Liquidity = mean(STOM, STOQ, STOA)

---

### Barra 因子暴露计算器

**类名：** `BarraCNE6Calculator`

**核心方法：**

```python
from barra_calculator import BarraCNE6Calculator
from enum.configs import PreprocessConfig, FactorConfig

# 1. 创建计算器
calc = BarraCNE6Calculator(
    start_date='20200101',
    end_date='20251231',
    preprocess_cfg=PreprocessConfig(),
    factor_cfg=FactorConfig()
)

# 2. 加载数据
calc.data = pd.read_pickle('barra_test_data.pkl')

# 3. 计算因子暴露
factor_df = calc.calculate_factors(
    do_winsorize=True,  # 去极值
    do_neutralize=False,  # 中性化
    do_standardize=True  # 标准化
)

# 4. 获取因子数据
factor_list = calc.factory.get_factor_list()
# ['size', 'momentum', 'dividend_yield', 'value',
#  'quality', 'volatility', 'growth', 'liquidity']

# 5. 保存结果
calc.save_results('factor_exposure.pkl')
```

**预处理配置参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| winsorize_method | 'mad' | 去极值方法（mad/percentile/sigma） |
| winsorize_n | 3.0 | 去极值阈值 |
| neutralize_method | 'industry' | 中性化方法（industry/market/both） |
| standardize_method | 'zscore' | 标准化方法（zscore/rank） |

---

### 因子收益率计算器

**核心原理：**

使用加权最小二乘回归（WLS）计算因子收益率：

```
R_i = α + Σ β_k × X_k + Σ γ_j × I_j + ε

其中：
- R_i：股票超额收益
- X_k：风格因子暴露
- I_j：行业哑变量
- 权重：w = exp(ln(市值))
```

**约束条件：**

行业因子收益率加权平均为0（市值加权）：
```
Σ (行业市值占比 × 行业因子收益率) = 0
```

**核心函数：**

```python
from barra_factor_return_calculator import (
    calculate_constrained_weighted_factor_returns,
    calculate_barra_factor_return
)

# 单日因子收益率
returns = calculate_constrained_weighted_factor_returns(
    df,
    date='20201231',
    style_factors=['size', 'momentum', 'value', 'quality'],
    industry_column='industry',
    size_factor_column='size',
    excess_return_column='excess_return'
)

# 全量因子收益率序列
returns_df = calculate_barra_factor_return(
    factor_df,
    style_factors_to_use=['size', 'momentum', 'value', 'quality'],
    industry_col='industry',
    size_col='size',
    excess_return_col='price_return'
)
```

**返回结果：**

| 列名 | 说明 |
|------|------|
| size | 尺寸因子收益率 |
| momentum | 动量因子收益率 |
| value | 价值因子收益率 |
| quality | 质量因子收益率 |
| volatility | 波动率因子收益率 |
| growth | 成长因子收益率 |
| liquidity | 流动性因子收益率 |
| dividend_yield | 红利因子收益率 |
| industry_* | 行业因子收益率 |

---

## 性能优化

项目针对大规模数据（千万行级别）进行了向量化优化：

| 操作 | 原耗时 | 优化后耗时 |
|------|--------|------------|
| 去极值 | 300秒 | 10秒 |
| 中性化 | 500秒 | 5-15秒 |
| 标准化 | 60秒 | 2-5秒 |

**优化策略：**
- 使用 `groupby.transform` 替代 `groupby.apply`
- 使用 `numpy.einsum` 加速矩阵运算
- 批量处理多列，减少 groupby 开销
- 禁用排序（`sort=False`）提升 20-30% 性能

---

## 数据源配置

支持多种数据加载方式：

```python
from barra_raw_data import BarraRawData, BarraSQLTemplates

# 从 pickle 文件加载
loader.load_raw_data_from_pickle('data.pkl')

# 从 feather 文件加载
loader.load_raw_data_from_feather('data.feather')

# 从 SQL 数据库加载
query = BarraSQLTemplates.get_barra_data('20200101', '20251231')
loader.load_raw_data_from_sql(query, conn)

# 行业数据
industry_query = BarraSQLTemplates.get_industry_data_query()
loader.load_industry_data_from_sql(industry_query, conn)
```

---

## 维护与扩展

- **添加新因子：** 实现新的因子子类，继承 `BarraBaseFunctions`，注册到 `BarraFactorFactory` 的 factors 字典中。
- **变更计算流程：** 修改 `BarraCNE6Calculator` 或 `barra_factor_return_calculator.py` 中的计算逻辑。
- **添加单元测试：** 在 `barra_model_test.py` 中编写测试用例，覆盖数据加载、因子计算及收益率计算等流程。

---

## 参考文献

- Barra CNE6 模型文档
- MSCI Barra Risk Model Handbook