"""
Barra CNE6 假数据生成与测试用例
===========================
基于 BarraSQLTemplates.get_barra_data 方法生成的假数据
数量级：约 2000 只股票 × 1500 个交易日 ≈ 300 万行
"""

import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
from typing import Tuple
from app.model.v5.data.fund_style_and_return_data import get_factor_returns, get_residuals


N_STOCKS = 2000
N_DAYS = 1500
START_DATE = datetime(2020, 1, 1)


def generate_stock_codes(n: int = N_STOCKS) -> list:
    """生成股票代码列表（6 位数字）"""
    return [f"{i:06d}" for i in range(1, n + 1)]


def generate_trade_dates(n: int = N_DAYS, start: datetime = START_DATE) -> list:
    """生成交易日期列表（跳过周末）"""
    dates = []
    current = start
    while len(dates) < n:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def generate_barra_data(
    n_stocks: int = N_STOCKS,
    n_days: int = N_DAYS,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成 Barra 假数据（基于 get_barra_data 方法的字段）

    字段说明：
    - busi_date: 交易日期
    - scr_code: 股票代码
    - market_value: 总市值
    - pct_change: 个股日收益率
    - close: 收盘价
    - index_return: 指数收益率
    - pred_divd_rate: 预测股息率
    - divd_rate_ttm: 股息率 TTM
    - bp: 账面市值比 (1/PB)
    - net_prof_ttm: 净利润 TTM
    - ebit_ttm: EBIT TTM
    - corp_val_crcp: 企业价值
    - pred_pera_avg: 分析师预测 E/P
    - total_liability: 总负债
    - total_asset: 总资产
    - total_income_ttm: 营业收入 TTM
    - total_cost_ttm: 营业成本 TTM
    - cash_flow_ttm: 经营性现金流 TTM
    - pred_eps_std: 预测 EPS 标准差
    - circulate_shares: 流通股本
    - capex: 资本支出
    - accr_bs: 应计项目
    - pred_net_prof_gtrt: 预测净利润增长率
    - income_ps: 每股营业收入
    - eps: 每股收益
    - mth_tnrt: 月换手率
    - quar_tnrt: 季换手率
    - ann_tnrt: 年换手率

    Returns
    -------
    pd.DataFrame
        生成的假数据 DataFrame
    """
    np.random.seed(seed)

    stock_codes = generate_stock_codes(n_stocks)
    trade_dates = generate_trade_dates(n_days, START_DATE)

    n_rows = n_stocks * n_days
    print(f"生成数据：{n_stocks} 股票 × {n_days} 交易日 = {n_rows:,} 行")

    df = pd.DataFrame({
        'secu_code': np.repeat(stock_codes, n_days),
        'end_date': np.tile([d.strftime('%Y%m%d') for d in trade_dates], n_stocks),
    })

    df['rept_data_date'] = pd.to_datetime(df['end_date'].astype(str)) + pd.DateOffset(days=1) - pd.offsets.QuarterEnd()
    df['rept_data_date'] = df['rept_data_date'].astype(str).str.replace('-', '').astype(int)
    df['market_value'] = np.random.lognormal(10, 1, n_rows) * 1e8

    close_base = np.random.uniform(5, 100, n_rows)
    pct_change = np.random.normal(0, 0.02, n_rows)
    df['close'] = close_base
    df['pct_change'] = pct_change
    df['prev_close'] = close_base / (1 + pct_change)
    df['prev_close'] = df['prev_close'].replace(0, 1)

    index_return = pd.Series(index=[d.strftime('%Y%m%d') for d in trade_dates], data=np.random.normal(0, 0.01, len(trade_dates))).reset_index()
    index_return.columns = ['end_date', 'index_return']
    df = df.merge(index_return, on=['end_date'], how='inner')
    # df['index_return'] = np.random.normal(0, 0.01, n_rows)

    df['pred_divd_rate'] = np.random.uniform(0, 0.05, n_rows)
    df['divd_rate_ttm'] = np.random.uniform(0, 0.05, n_rows)

    pb = np.random.uniform(0.5, 50, n_rows)
    df['bp'] = 1 / pb
    df['pb'] = pb

    df['net_profit_ttm'] = np.random.lognormal(8, 2, n_rows) * 1e7
    df['ebit_ttm'] = np.random.lognormal(8.5, 2, n_rows) * 1e7
    df['corp_val_crcp'] = df['market_value'] * np.random.uniform(1, 3, n_rows)
    df['pred_pera_avg'] = np.random.uniform(0, 0.2, n_rows)

    df['total_liability'] = np.random.lognormal(9, 2, n_rows) * 1e7
    df['total_asset'] = df['total_liability'] + np.random.lognormal(9.5, 2, n_rows) * 1e7

    df['total_income_ttm'] = np.random.lognormal(9, 2, n_rows) * 1e8
    df['total_cost_ttm'] = df['total_income_ttm'] * np.random.uniform(0.6, 0.95, n_rows)
    df['cash_flow_ttm'] = np.random.lognormal(8, 2, n_rows) * 1e7

    df['pred_eps_std'] = np.random.uniform(0, 0.5, n_rows)
    df['circulate_shares'] = np.random.lognormal(8, 1, n_rows) * 1e7
    df['capex'] = np.random.lognormal(7, 2, n_rows) * 1e6

    df['accr_bs'] = df['net_profit_ttm'] - df['cash_flow_ttm']

    df['pred_net_prof_gtrt'] = np.random.normal(0, 0.2, n_rows)
    df['income_ps'] = np.random.lognormal(1, 0.5, n_rows) * 10
    df['eps'] = np.random.normal(0.5, 0.3, n_rows)

    df['mth_tnrt'] = np.random.uniform(0, 2, n_rows)
    df['quar_tnrt'] = np.random.uniform(0, 5, n_rows)
    df['ann_tnrt'] = np.random.uniform(0, 10, n_rows)

    for col in df.columns:
        if col not in ['secu_code', 'end_date', 'index_return', 'rept_data_date', 'pct_change']:
            if df[col].dtype in [np.float64, np.float32]:
                nan_ratio = 0.01
                nan_idx = np.random.choice(n_rows, size=int(n_rows * nan_ratio), replace=False)
                df.loc[nan_idx, col] = np.nan

    print(f"数据生成完成，形状：{df.shape}")
    return df


def generate_industry_data(n_stocks: int = N_STOCKS, seed: int = 42) -> pd.DataFrame:
    """生成行业分类假数据"""
    np.random.seed(seed)
    stock_codes = generate_stock_codes(n_stocks)

    industry_l1 = ['银行', '电子', '医药', '食品饮料', '计算机', '房地产', '化工', '机械设备']
    industry_l2 = [f"{ind}_子行业{i}" for ind in industry_l1 for i in range(1, 4)]

    return pd.DataFrame({
        'secu_code': stock_codes,
        'secu_name': [f"股票{i:04d}" for i in range(1, n_stocks + 1)],
        'industry_code_l1': np.random.choice(industry_l1, n_stocks),
        'industry_name_l1': np.random.choice(industry_l1, n_stocks),
        'industry_code_l2': np.random.choice(industry_l2, n_stocks),
        'industry_name_l2': np.random.choice(industry_l2, n_stocks),
    })


class TestBarraDataGeneration:
    """Barra 数据生成测试类"""

    def test_generate_data_shape(self):
        """测试数据形状"""
        df = generate_barra_data(n_stocks=100, n_days=10)
        expected_rows = 100 * 10
        assert len(df) == expected_rows, f"期望 {expected_rows} 行，实际 {len(df)} 行"
        print(f"[PASS] 数据形状: {df.shape}")

    def test_generate_data_columns(self):
        """测试必要字段存在"""
        df = generate_barra_data(n_stocks=100, n_days=10)
        required_cols = [
            'secu_code', 'end_date', 'market_value', 'pct_change', 'close',
            'index_return', 'pred_divd_rate', 'divd_rate_ttm', 'bp',
            'net_profit_ttm', 'ebit_ttm', 'corp_val_crcp', 'pred_pera_avg',
            'total_liability', 'total_asset', 'total_income_ttm',
            'total_cost_ttm', 'cash_flow_ttm', 'pred_eps_std',
            'circulate_shares', 'capex', 'accr_bs', 'pred_net_prof_gtrt',
            'income_ps', 'eps', 'mth_tnrt', 'quar_tnrt', 'ann_tnrt'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        assert not missing, f"缺少字段: {missing}"
        print(f"[PASS] 必要字段完整，共 {len(df.columns)} 列")

    def test_data_range(self):
        """测试数据值范围合理"""
        df = generate_barra_data(n_stocks=100, n_days=10)

        assert df['market_value'].min() > 0, "市值需大于 0"
        assert df['pct_change'].std() > 0, "收益率应有波动"
        assert df['total_asset'].min() > 0, "总资产需大于 0"
        assert df['total_liability'].min() > 0, "总负债需大于 0"

        print(f"[PASS] 数据值范围合理")
        print(f"  - 市值: {df['market_value'].min():.2e} ~ {df['market_value'].max():.2e}")
        print(f"  - 收益率均值: {df['pct_change'].mean():.4f}, 标准差: {df['pct_change'].std():.4f}")
        print(f"  - 总资产: {df['total_asset'].min():.2e} ~ {df['total_asset'].max():.2e}")

    def test_industry_data(self):
        """测试行业数据生成"""
        ind_df = generate_industry_data(n_stocks=100)
        assert len(ind_df) == 100, "行业数据行数不对"
        assert 'industry_name_l1' in ind_df.columns, "缺少行业字段"
        print(f"[PASS] 行业数据生成正常: {ind_df.shape}")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Barra 假数据生成测试")
    print("=" * 60)

    tester = TestBarraDataGeneration()
    tester.test_generate_data_shape()
    tester.test_generate_data_columns()
    tester.test_data_range()
    tester.test_industry_data()

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


# ============================================================================
# 实际数据取数函数
# ============================================================================
from app.model.v5.cne6.const import DATA_DIR
# DATA_DIR = r'D:\claude\Barra\data\barra_test'


def load_real_index_data() -> pd.DataFrame:
    """
    加载实际指数行情数据 (d_s)

    数据来源: dws_var_info_indx_quot_dd.feather

    Returns
    -------
    pd.DataFrame
        指数行情数据，包含 end_date, close, index_return 等字段
    """
    file_path = os.path.join(DATA_DIR, 'dws_var_info_indx_quot_dd.feather')
    df = pd.read_feather(file_path)
    df.columns = ['end_date', 'index_close', 'index_return']
    print(f"[指数行情] 加载完成: {df.shape}")
    return df


def load_real_financial_data() -> pd.DataFrame:
    """
    加载实际财务数据和市场数据 (a_s)

    数据来源: dm_inrs_ins_fin_indx_dd_*.feather 多个文件合并

    Returns
    -------
    pd.DataFrame
        财务数据和行情数据，筛选 6/3/0 开头的股票
    """
    pattern = os.path.join(DATA_DIR, 'dm_inrs_ins_fin_indx_dd_*.feather')
    file_list = glob(pattern)

    df_list = []
    for file in file_list:
        df_list.append(pd.read_feather(file))

    df = pd.concat(df_list)

    # 筛选股票代码以 6/3/0 开头
    df = df[(df['secu_code'].str.startswith('6')) |
            (df['secu_code'].str.startswith('3')) |
            (df['secu_code'].str.startswith('0'))]
    
    df = df.sort_values(['secu_code', 'end_date', 'rept_data_date']).drop_duplicates(['secu_code', 'end_date'], keep='last')

    print(f"[财务+行情] 加载完成: {df.shape}, 共 {len(file_list)} 个文件")
    return df


def load_real_prediction_data() -> pd.DataFrame:
    """
    加载实际预测数据 (b_s)

    数据来源: dm_inrs_stk_prof_cmph_perd_dd.feather

    Returns
    -------
    pd.DataFrame
        预测数据，筛选 6/3/0 开头的股票
    """
    file_path = os.path.join(DATA_DIR, 'dm_inrs_stk_prof_cmph_perd_dd.feather')
    df = pd.read_feather(file_path)

    # 筛选股票代码以 6/3/0 开头
    df = df[(df['secu_code'].str.startswith('6')) |
            (df['secu_code'].str.startswith('3')) |
            (df['secu_code'].str.startswith('0'))]

    print(f"[预测数据] 加载完成: {df.shape}")
    return df


def load_real_financial_data_test(start) -> pd.DataFrame:
    """
    加载实际财务数据和市场数据 (a_s)

    数据来源: dm_inrs_ins_fin_indx_dd_*.feather 多个文件合并

    Returns
    -------
    pd.DataFrame
        财务数据和行情数据，筛选 6/3/0 开头的股票
    """
    pattern = os.path.join(DATA_DIR, 'dm_inrs_ins_fin_indx_dd_*.feather')
    file_list = glob(pattern)

    df_list = []
    for file in file_list:
        if file.split('_')[-2] < start:
            continue
        df_list.append(pd.read_feather(file))

    df = pd.concat(df_list)

    # 筛选股票代码以 6/3/0 开头
    df = df[(df['secu_code'].str.startswith('6')) |
            (df['secu_code'].str.startswith('3')) |
            (df['secu_code'].str.startswith('0'))]

    df = df.sort_values(['secu_code', 'end_date', 'rept_data_date']).drop_duplicates(['secu_code', 'end_date'], keep='last')

    print(f"[财务+行情] 加载完成: {df.shape}, 共 {len(file_list)} 个文件")
    return df


def load_real_prediction_data_test(start) -> pd.DataFrame:
    """
    加载实际预测数据 (b_s)

    数据来源: dm_inrs_stk_prof_cmph_perd_dd.feather

    Returns
    -------
    pd.DataFrame
        预测数据，筛选 6/3/0 开头的股票
    """
    pattern = os.path.join(DATA_DIR, 'dm_inrs_stk_prof_cmph_perd_dd_*.feather')
    file_list = glob(pattern)

    df_list = []
    for file in file_list:
        if file.split('_')[-2] < start:
            continue
        df_list.append(pd.read_feather(file))

    df = pd.concat(df_list)
    df = df.rename(columns={'scr_code': 'secu_code'})

    # 筛选股票代码以 6/3/0 开头
    df = df[(df['secu_code'].str.startswith('6')) |
            (df['secu_code'].str.startswith('3')) |
            (df['secu_code'].str.startswith('0'))]

    print(f"[预测数据] 加载完成: {df.shape}")
    return df


def load_real_barra_data_test(start) -> pd.DataFrame:
    """
    加载并合并所有实际数据

    合并逻辑:
    1. a_s (财务+行情) 和 b_s (预测) 按 secu_code + end_date 合并
    2. 过滤掉 corp_val_crcp 为空的记录
    3. 重命名列名以匹配 Barra 模型要求

    Returns
    -------
    pd.DataFrame
        完整的 Barra 实际数据
    """
    # 加载三类数据
    a_s = load_real_financial_data_test(start)
    b_s = load_real_prediction_data_test(start)
    d_s = load_real_index_data()

    # 合并财务数据和预测数据
    c_s = b_s.merge(a_s, on=['secu_code', 'end_date'], how='right')

    # 重命名列名
    c_s.columns = ['end_date', 'secu_code', 'pred_divd_rate', 'pred_pera_avg',
                   'pred_eps_std', 'pred_net_prof_gtrt', 'rela_date', 'rept_data_date',
                   'ins_num', 'net_profit_ttm', 'ebit_ttm', 'total_liability', 'total_asset',
                   'total_income_ttm', 'total_cost_ttm', 'cash_flow_ttm', 'capex',
                   'accr_bs', 'income_ps', 'eps', 'market_value', 'pct_change', 'close',
                   'divd_rate_ttm', 'bp', 'corp_val_crcp', 'circulate_shares', 'mth_tnrt',
                   'quar_tnrt', 'ann_tnrt']

    # 合并指数行情数据
    final_df = c_s.merge(d_s, on='end_date', how='left')

    print(f"[完整数据] 合并完成: {final_df.shape}")
    print(f"  - 股票数: {final_df['secu_code'].nunique()}")
    print(f"  - 交易日数: {final_df['end_date'].nunique()}")

    return final_df


def load_real_barra_data() -> pd.DataFrame:
    """
    加载并合并所有实际数据

    合并逻辑:
    1. a_s (财务+行情) 和 b_s (预测) 按 secu_code + end_date 合并
    2. 过滤掉 corp_val_crcp 为空的记录
    3. 重命名列名以匹配 Barra 模型要求

    Returns
    -------
    pd.DataFrame
        完整的 Barra 实际数据
    """
    # 加载三类数据
    a_s = load_real_financial_data()
    b_s = load_real_prediction_data()
    d_s = load_real_index_data()

    # 合并财务数据和预测数据
    c_s = b_s.merge(a_s, on=['secu_code', 'end_date'], how='right')

    # 重命名列名
    c_s.columns = ['end_date', 'secu_code', 'pred_divd_rate', 'pred_pera_avg',
                   'pred_eps_std', 'pred_net_prof_gtrt', 'rela_date', 'rept_data_date',
                   'ins_num', 'net_profit_ttm', 'ebit_ttm', 'total_liability', 'total_asset',
                   'total_income_ttm', 'total_cost_ttm', 'cash_flow_ttm', 'capex',
                   'accr_bs', 'income_ps', 'eps', 'market_value', 'pct_change', 'close',
                   'divd_rate_ttm', 'bp', 'corp_val_crcp', 'circulate_shares', 'mth_tnrt',
                   'quar_tnrt', 'ann_tnrt']

    # 合并指数行情数据
    final_df = c_s.merge(d_s, on='end_date', how='left')

    print(f"[完整数据] 合并完成: {final_df.shape}")
    print(f"  - 股票数: {final_df['secu_code'].nunique()}")
    print(f"  - 交易日数: {final_df['end_date'].nunique()}")

    return final_df


def load_real_industry_data(raw_df) -> pd.DataFrame:
    """
    加载实际行业分类数据

    数据来源: 如果存在行业数据文件则加载，否则从财务数据中提取股票列表生成默认行业分类

    Returns
    -------
    pd.DataFrame
        行业分类数据，包含 secu_code, industry_name_l1 等字段
    """
    # 尝试加载行业数据文件
    industry_file = os.path.join(DATA_DIR, 'industry.feather')
    if os.path.exists(industry_file):
        df = pd.read_feather(industry_file)
        df.columns = ['secu_code', 'industry']
        print(f"[行业数据] 加载完成: {df.shape}")
        return df

    # 如果不存在行业文件，从财务数据中提取股票列表生成默认行业分类
    print("[行业数据] 未找到行业数据文件，生成默认行业分类...")
    stock_codes = raw_df['secu_code'].unique()

    # 默认行业分类
    industry_l1 = ['银行', '电子', '医药', '食品饮料', '计算机', '房地产', '化工', '机械设备']

    df = pd.DataFrame({
        'secu_code': stock_codes,
        'secu_name': [f"股票_{code}" for code in stock_codes],
        'industry_code_l1': np.random.choice(industry_l1, len(stock_codes)),
        'industry_name_l1': np.random.choice(industry_l1, len(stock_codes)),
    })

    print(f"[行业数据] 生成完成: {df.shape}")
    return df


def load_factor_data(start) -> pd.DataFrame:
    pattern = os.path.join(DATA_DIR, 'factors_*.feather')
    file_list = glob(pattern)

    df_list = []
    for file in file_list:
        df_list.append(pd.read_feather(file))

    df = pd.concat(df_list).sort_values(['end_date', 'secu_code']).reset_index(drop=True)
    print(f"[因子暴露数据] 加载完成: {df.shape}, 共 {len(file_list)} 个文件")
    return df


def load_factor_returns() -> pd.DataFrame:
    df = get_factor_returns()
    df['end_date'] = df['end_date'].astype(str)
    print(f"[因子收益率数据] 加载完成: 共{df.shape[0]}个交易日")
    return df


def load_residuals() -> pd.DataFrame:
    df = get_residuals()
    print(f"[因子收益率数据] 加载完成: 共{df.shape[0]}个交易日")
    return df

if __name__ == "__main__":
    df = load_factor_returns()
    run_tests()

    print("\n生成完整测试数据...")
    print("-" * 60)

    df = generate_barra_data()
    print(f"\n最终数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(df.head())

    industry_df = generate_industry_data()
    print(f"\n行业数据形状: {industry_df.shape}")
    print(industry_df.head())

    output_data = input("是否保存数据到文件？(y/n): ").strip().lower()
    if output_data == 'y':
        df.to_pickle('barra_test_data.pkl')
        industry_df.to_pickle('barra_industry_test_data.pkl')
        print("数据已保存!")
