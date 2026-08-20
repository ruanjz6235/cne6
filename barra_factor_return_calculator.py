import sys

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from arch import arch_model
from joblib import Parallel, delayed
from app.utils.DbHandleUtil import DbHandleUtilNew
from app.model.v5.const import *
from app.model.v5.cne6.barra_storage import BarraStorage

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


def calculate_constrained_weighted_factor_returns(df, last, date, style_factors, industry_column, size_factor_column,
                                                  excess_return_column):
    """
    计算日度因子收益率。

    参数：
        df (pd.DataFrame): 包含每日股票数据（包括超额收益、预处理后的风格因子暴露、
                           行业分类和规模因子）的 DataFrame。
        date (str 或 datetime): 要计算因子收益率的特定日期。
        style_factors (list): 对应风格因子列名的列表。
        industry_column (str): 行业分类的列名。
        size_factor_column (str): 规模因子（对数市值）的列名。
        excess_return_column (str): 股票超额收益的列名。

    返回：
        pd.Series: 包含计算出的日度因子收益率的 Series。
                   如果给定日期的数据不足，则返回 None。
    """

    df_daily = df[df['end_date'] == last].dropna().set_index('secu_code')
    df_date = df[df['end_date'] == date].dropna().set_index('secu_code')
    codes = df_daily.index.intersection(df_date.index)
    df_daily = df_daily[df_daily.index.isin(list(codes))]
    df_date = df_date[df_date.index.isin(codes)]

    for col in df.columns[~np.isin(df.columns, ['secu_code', 'end_date', 'industry'])]:
        df_daily_no_zero = df_daily[df_daily[col] != 0]
        if df_daily_no_zero.empty:
            print(f"No valid data available for {col} and {date}")
            return None, None, None, None, None, None, None

    # 因变量股票超额收益
    R = df_date[excess_return_column].values

    # 自变量因子暴露
    industry_dummies = pd.get_dummies(df_daily[industry_column], dtype=int)
    industry_dummies = industry_dummies.rename(columns=ind_dict)
    X_style = df_daily[style_factors].values
    X_industry = industry_dummies.values

    # 截距
    intercept = np.ones((len(df_daily), 1))

    # 暴露矩阵
    X = np.hstack((intercept, X_style, X_industry))

    # 因子名称
    factor_names = ['alpha'] + style_factors + industry_dummies.columns.tolist()

    # 权重
    weights = np.exp(df_daily[size_factor_column].values)
    weights[weights <= 0] = 1e-6  # 将非正权重替换为一个小的正数
    W_diag = np.diag(weights)  # 权重对角矩阵

    # 加权最小二乘的目标函数:最小化 ||W^(1/2) * (R - Xb)||^2 = (R - Xb)^T * W * (R - Xb)
    def objective(b):
        return (R - X @ b).T @ W_diag @ (R - X @ b)

    # 约束条件
    industry_market_caps = df_daily.groupby(industry_column)[size_factor_column].apply(lambda x: np.exp(x).sum())
    total_market_cap = industry_market_caps.sum()
    industry_weights_for_constraint = industry_market_caps / total_market_cap

    num_factors = len(factor_names)
    A_constraint = np.zeros(num_factors)

    for ind_name, weight_val in industry_weights_for_constraint.items():
        dummy_col_name = ind_dict[ind_name]
        if dummy_col_name in factor_names:
            idx = factor_names.index(dummy_col_name)
            A_constraint[idx] = weight_val

    constraints = ({'type': 'eq', 'fun': lambda b: A_constraint @ b - 0})
    b0 = np.zeros(num_factors)

    result = minimize(objective, b0, constraints=constraints)

    if result.success:
        factor_returns = pd.Series(result.x, index=factor_names, name=date)

        # ===== 补充：计算5个回归补充指标 =====
        # (1) 因子模型估计的个股收益率
        pred_returns = X @ factor_returns.values

        # (2) 残差
        residuals = R - pred_returns

        # (3) R方 (加权R-squared)
        ssr = (residuals ** 2 * weights).sum()
        weighted_mean_R = (R * weights).sum() / weights.sum()
        sst = ((R - weighted_mean_R) ** 2 * weights).sum()
        r_squared = 1 - ssr / sst

        # (4) 调整R方
        n = len(R)
        k = len(factor_names)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)

        # (5) 模型t值
        residual_variance = ssr / (n - k)
        XWX_inv = np.linalg.pinv(X.T @ W_diag @ X)
        factor_returns_cov = residual_variance * XWX_inv
        factor_returns_se = np.sqrt(np.diag(factor_returns_cov))
        t_stat = pd.Series(factor_returns.values / factor_returns_se, index=factor_names, name=date)
        # ===== 补充结束 =====
        exposure = df_daily.copy()
        exposure['residual'] = residuals
        exposure['pred_return'] = pred_returns

        return factor_returns, pred_returns, residuals, r_squared, adj_r_squared, t_stat, exposure.reset_index()
    else:
        print(f"Optimization failed for {date}: {result.message}")
        return None, None, None, None, None, None, None


def calculate_barra_factor_return(barra, style_factors_to_use=None,
                                  industry_col='industry', size_col='size', excess_return_col='price_return'):
    if style_factors_to_use is None:
        style_factors_to_use = ['size', 'volatility', 'momentum', 'value', 'dividend_yield', 'quality', 'growth', 'liquidity']

    factor_returns, pred_returns, residuals, r_squared, adj_r_squared, t_stat = {}, {}, {}, {}, {}, {}
    dates = barra['end_date'].unique()
    for last_date, current_date in zip(dates[:-1], dates[1:]):
        if current_date < '20160120':
            continue
        print(f"\nCalculating Barra factor returns for {current_date}...")
        fr, pr, res, r2, adj_r2, ts, exposure = calculate_constrained_weighted_factor_returns(
            barra,
            last_date,
            current_date,
            style_factors_to_use,
            industry_col,
            size_col,
            excess_return_col
        )
        if fr is not None:
            DbHandleUtilNew.save(df=exposure, table_name='barra_exposure', bind='zhijunfund')
            factor_returns[current_date] = fr
            r_squared[current_date] = r2
            adj_r_squared[current_date] = adj_r2
            t_stat[current_date] = ts

    # 提取各指标为DataFrame/Series
    factor_returns = pd.DataFrame(factor_returns).T
    t_stat = pd.DataFrame(t_stat).T
    r_squared = pd.DataFrame(pd.Series(r_squared, name='r2'))
    adj_r_squared = pd.DataFrame(pd.Series(adj_r_squared, name='adj_r2'))
    beta = pd.concat([factor_returns, r_squared, adj_r_squared], axis=1).reset_index().rename(columns={'index': 'end_date'})
    DbHandleUtilNew.save(df=beta, table_name='barra_beta', bind='zhijunfund')


def determine_calc_factor_return_range(data) -> tuple:
    """确定计算时段"""
    dates = data['end_date'].drop_duplicates().sort_values().tolist()
    source_start_date, source_end_date = dates[0], dates[-1]
    storage = BarraStorage()
    latest_exposure_date = storage.get_latest_factor_return_date()

    if latest_exposure_date:
        if latest_exposure_date >= dates[-1]:
            print(f"因子收益率数据已是最新: latest={latest_exposure_date}, source_end={source_end_date}")
            return None, None, None, False
        # 增量模式
        calc_start = dates[dates.index(latest_exposure_date) + 1]
        calc_end = source_end_date

        print(f"[增量模式] 计算时段: {calc_start} → {calc_end}")
        return latest_exposure_date, calc_start, calc_end, True
    else:
        # 全量模式
        print(f"[全量模式] 计算时段: {source_start_date} → {source_end_date}")
        return None, None, source_end_date, True


# ===== 补充：3个向量化预测函数 =====

def factor_return_vol_forecast_vectorized(factor_returns_df, tradingdays, n_jobs=-1):
    """
    向量化因子波动率预测 (ARX-GARCH(1,1))
    
    参数:
        factor_returns_df: DataFrame, 列=['TradingDay', factor1, factor2, ...]
        tradingdays: list, 需要预测的交易日列表
        n_jobs: int, 并行作业数, -1表示使用所有CPU核心
    
    返回:
        (PreBetaVol, PreBetaVol_5D, PreBetaVol_5p, forecasted_factor_return): 4个DataFrame
    """
    factor_cols = [c for c in factor_returns_df.columns if c != 'end_date']
    scale = 1000  # 数值稳定性缩放因子
    
    def _fit_single_factor(hist_series):
        """拟合单个因子的ARX-GARCH并返回预测结果"""
        y = hist_series.dropna().values * scale
        if len(y) < 252:
            return None, None, None, None, None, None, None, None
        
        try:
            am = arch_model(y, mean='ARX', lags=2, vol='GARCH', p=1, o=0, q=1, power=2.0, dist='Normal')
            res = am.fit(update_freq=5, disp='off')
            forecasts = res.forecast(horizon=5)
            
            # 波动率预测
            var_matrix = forecasts.variance.dropna().T.values
            vol_1d = np.sqrt(var_matrix[0, 0]) / scale
            vol_5d = np.sqrt(var_matrix.sum()) / scale
            vol_5p = np.sqrt(var_matrix) / scale
            v1, v2, v3, v4, v5 = vol_5p[:, 0]

            # ARX均值预测
            ar_params = res.params.iloc[0:3].copy()
            ar_pvalues = res.pvalues.iloc[0:3]
            ar_params[ar_pvalues > 0.1] = 0
            
            lags = hist_series.dropna().iloc[-2:].values * scale
            e_ret = (ar_params.iloc[0] + ar_params.iloc[1] * lags[1] + ar_params.iloc[2] * lags[0]) / scale
            
            return vol_1d, vol_5d, e_ret, v1, v2, v3, v4, v5
        except Exception:
            return None, None, None, None, None, None, None, None

    for tradingday in tradingdays:
        hist_data = factor_returns_df[factor_returns_df['end_date'] <= tradingday]
        # 并行处理所有因子 (向量化替代for循环)
        result = Parallel(n_jobs=n_jobs)(
            delayed(_fit_single_factor)(hist_data[ff]) for ff in factor_cols)
        result = pd.DataFrame(result, index=factor_cols).T
        result['pred_type'] = ['vol_1d', 'vol_5d', 'pred_return', 'vol_1st', 'vol_2nd', 'vol_3rd', 'vol_4th', 'vol_5th']
        result['end_date'] = tradingday
        DbHandleUtilNew.save(df=result, table_name='barra_beta_pred', bind='zhijunfund')


def determine_calc_pre_beta_vol_range(data) -> tuple:
    """确定计算时段"""
    dates = data['end_date'].drop_duplicates().sort_values().tolist()
    source_start_date, source_end_date = dates[0], dates[-1]
    storage = BarraStorage()
    latest_exposure_date = storage.get_latest_pre_beta_vol_date()
    # latest_exposure_date = 20260415

    if latest_exposure_date:
        if latest_exposure_date >= dates[-1]:
            print(f"因子波动率预测数据已是最新: latest={latest_exposure_date}, source_end={source_end_date}")
            return None, None, False
        # 增量模式
        calc_start = dates[dates.index(latest_exposure_date) + 1]
        calc_end = source_end_date

        print(f"[增量模式] 计算时段: {calc_start} → {calc_end}")
        return calc_start, calc_end, True
    else:
        # 全量模式
        print(f"[全量模式] 计算时段: {source_start_date} → {source_end_date}")
        return None, source_end_date, True


def factor_return_correlation_ewma_vectorized(factor_returns_df, start, halflife=504):
    """
    向量化因子相关系数预测 (EWMA)
    
    参数:
        factor_returns_df: DataFrame, 列=['TradingDay', factor1, factor2, ...]
        tradingdays: list, 需要预测的交易日列表
        halflife: int, EWMA半衰期, 默认504日(2年)
    
    返回:
        pd.DataFrame: EWMA相关系数矩阵, 列=['TradingDay', 'factor', factor1, factor2, ...]
    """
    factor_cols = [c for c in factor_returns_df.columns if c != 'end_date']
    ewma_corr = factor_returns_df[factor_cols].ewm(halflife=halflife).corr().reset_index()
    ewma_corr = ewma_corr.rename(columns={'level_0': 'end_date', 'level_1': 'factor'})
    ewma_corr['end_date'] = ewma_corr['end_date'].map(dict(enumerate(factor_returns_df['end_date'].tolist())))
    ewma_corr = ewma_corr[ewma_corr['end_date'] >= start]
    DbHandleUtilNew.save(df=ewma_corr, table_name='barra_beta_corr', bind='zhijunfund')


def residual_volatility_forecast_vectorized(resid_df, tradingdays, n_jobs=-1):
    """
    向量化个股残差波动率预测 (ARX-GARCH)
    
    参数:
        resid_df: DataFrame, index=dates, columns=codes, value=residual
        tradingdays: list, 需要预测的交易日列表
        n_jobs: int, 并行作业数
    
    返回:
        pd.DataFrame: 残差波动率预测, 列=['TradingDay', 'SecuCode', 'resid_vol_forecast']
    """
    stocks = resid_df.columns
    scale = 1000
    
    def _fit_single_stock(hist_series):
        """拟合单只个股残差的GARCH"""
        y = hist_series.dropna().values * scale
        if len(y) < 252:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        try:
            am = arch_model(y, mean='ARX', lags=2, vol='GARCH', p=1, o=0, q=1, power=2.0, dist='Normal')
            res = am.fit(disp='off')
            forecasts = res.forecast(horizon=5)
            # 波动率预测
            var_matrix = forecasts.variance.dropna().T.values
            vol_5d = np.sqrt(var_matrix.sum()) / scale
            vol_5p = np.sqrt(var_matrix) / scale
            v1, v2, v3, v4, v5 = vol_5p[:, 0]
            return v1, v2, v3, v4, v5, vol_5d
        except Exception:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    results = []
    for tradingday in tradingdays[251:]:
        print(tradingday)
        hist_data = resid_df.loc[:tradingday]
        # 并行处理所有股票 (向量化替代for循环)
        vols = Parallel(n_jobs=n_jobs)(delayed(_fit_single_stock)(hist_data[s]) for s in stocks)
        results.append(vols)

    index = pd.MultiIndex.from_product([tradingdays[251:], stocks])
    results_new = pd.DataFrame(np.array(results).reshape(len(stocks)*len(tradingdays[251:]), 6), index=index,
                               columns=['vol_1st', 'vol_2nd', 'vol_3rd', 'vol_4th', 'vol_5th', 'vol_5d']).dropna().reset_index()
    DbHandleUtilNew.save(df=results_new, table_name='barra_residual_pred', bind='zhijunfund')


def determine_calc_residual_vol_range(data) -> tuple:
    """确定计算时段"""
    dates = data.index.tolist()
    source_start_date, source_end_date = dates[0], dates[-1]
    storage = BarraStorage()
    latest_exposure_date = storage.get_latest_residual_vol_date()

    if latest_exposure_date:
        if latest_exposure_date >= dates[-1]:
            print(f"残差波动率预测数据已是最新: latest={latest_exposure_date}, source_end={source_end_date}")
            return None, None, False
        # 增量模式
        calc_start = dates[dates.index(latest_exposure_date) + 1]
        calc_end = source_end_date

        print(f"[增量模式] 计算时段: {calc_start} → {calc_end}")
        return calc_start, calc_end, True
    else:
        # 全量模式
        print(f"[全量模式] 计算时段: {source_start_date} → {source_end_date}")
        return None, source_end_date, True


def residual_volatility_forecast(resid_df, start, halflife=504):
    """
    个股残差波动率预测 - EWMA方法 (向量化)

    参数:
        resid_df: DataFrame, index=日期, columns=股票代码, value=残差
        tradingdays: list, 需要预测的交易日列表
        n_jobs: 保留参数，EWMA向量化实现不再需要并行，仅保持接口兼容
        halflife: float, EWMA半衰期，默认504

    返回:
        pd.DataFrame: 包含 'end_date', 'secu_code', 'vol_1st', 'vol_5d'
    """
    # 计算残差平方
    resid_sq = resid_df ** 2

    # EWMA条件方差预测（对t+1期方差的估计）
    # adjust=False 实现标准递归: v_t = (1-α)*r_{t-1}^2 + α*v_{t-1}
    ewma_var = resid_sq.ewm(halflife=halflife).mean()
    ewma_vol = np.sqrt(ewma_var)  # 预测波动率（年化需额外处理，此处保持日度）

    # 转换为长格式：MultiIndex (date, stock) -> value
    vol_long = ewma_vol.stack()
    vol_long.name = 'vol_1st'
    result = vol_long.reset_index()
    result.rename(columns={'level_0': 'end_date', 'level_1': 'secu_code'}, inplace=True)
    result['vol_5d'] = np.sqrt(5) * result['vol_1st']  # 5日累计波动率
    # 丢弃历史数据不足的股票（对应NaN行）
    result = result[result['end_date'] >= start]
    result.dropna(inplace=True)
    # 保存到数据库（表名与原GARCH版本一致）
    DbHandleUtilNew.save(df=result, table_name='barra_residual_pred', bind='zhijunfund')
