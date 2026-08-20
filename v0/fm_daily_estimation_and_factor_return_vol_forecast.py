#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:57:27 2019

@author: yunongwu
"""
import numpy as np
import pandas as pd
import datetime as dt
import statsmodels.api as sm
import time
from arch import arch_model
from app.common.log import logger
from app.utils import *

# def construct_db_connections():
#     jy_conn = DbUtil.get_conn(bind='jydb')
#     zj_conn = DbUtil.get_conn(bind='zj_data')
#     fd_conn = DbUtil.get_conn(bind='funddata')
#     return jy_conn, zj_conn

#%%
def get_last_run_date():
    query = """
    SELECT max(TradingDay) TradingDay FROM FM_WLS_Beta
    """
    last_run_date = pd.read_sql(query, DbUtil.get_conn('zj_data'))
    last_run_date = last_run_date.TradingDay.tolist()[0]
    return last_run_date


def get_available_date():
    query = """
    SELECT max(date) TradingDay FROM stock_daily_quote
    """
    last_available_date = pd.read_sql(query, DbUtil.get_conn('funddata'))
    last_available_date = last_available_date.TradingDay.tolist()[0]
    return last_available_date


def get_stock_price(start, end):
    query = """
    SELECT code SecuCode, date TradingDay, close Price
    FROM stock_daily_quote
    WHERE date > '%s' AND date <= '%s'
    """ % (start, end)
    stocks_price = pd.read_sql(query, DbUtil.get_conn('funddata'))
    stocks_price = stocks_price[~stocks_price['SecuCode'].str.startswith('688')]
    # Return data and fundamentals are merged by year and quarter, because fundamentals
    # are provided quarterly.
    stocks_price['TradingDay'] = pd.to_datetime(stocks_price['TradingDay'], errors='coerce')
    stocks_price['year'] = stocks_price['TradingDay'].dt.year
    stocks_price['quarter'] = stocks_price['TradingDay'].dt.quarter
    return stocks_price


def get_historical_stock_price(start, stock_list):
    historical_stocks_price = list()
    for code in stock_list:
        query = """
        SELECT code SecuCode, date TradingDay, close Price
        FROM stock_daily_quote
            WHERE code = '%s' AND date <= '%s'
            ORDER BY date desc
            limit 525
        """ % (code, start)
        historical_stock_price = pd.read_sql(query, DbUtil.get_conn('funddata'))
        historical_stocks_price.append(historical_stock_price)
    historical_stocks_price = pd.concat(historical_stocks_price, ignore_index=True, sort=False)
    historical_stocks_price['TradingDay'] = pd.to_datetime(historical_stocks_price['TradingDay'], errors='coerce')
    historical_stocks_price['year'] = historical_stocks_price['TradingDay'].dt.year
    historical_stocks_price['quarter'] = historical_stocks_price['TradingDay'].dt.quarter
    return historical_stocks_price
#%%

def get_mv_tr(stock_list, start, end):
    # QT_Performance still contains stock information during suspension.
    # TurnoverRate != 0 is a condition to confirm the stock is active on that day.
    query = """
    SELECT SecuCode, InnerCode
    FROM SecuMain
    WHERE SecuCode in %s
    AND SecuCategory = 1
    """ % (str(tuple(stock_list)))
    innercodes = pd.read_sql(query, DbUtil.get_conn('jydb'))
    innercode_list = innercodes.InnerCode.tolist()

    query = """
    SELECT
        InnerCode, TradingDay,
        TurnoverRate, NegotiableMV as NMV
    FROM QT_Performance
        WHERE InnerCode in %s
        AND TurnoverRate != 0
        AND TradingDay > '%s' AND TradingDay <= '%s'
    """ % (str(tuple(innercode_list)), start, end)
    mv_tr = pd.read_sql(query, DbUtil.get_conn('jydb'))
    mv_tr = pd.merge(mv_tr, innercodes, on='InnerCode', how='inner')
    mv_tr = mv_tr[['SecuCode', 'TradingDay', 'TurnoverRate', 'NMV']]
    DbHandleUtil.save('FM_mv_tr', mv_tr, 'zj_data', has_jsid=False)
    # values = list(zip(mv_tr['SecuCode'], mv_tr['TradingDay'], mv_tr['TurnoverRate'], mv_tr['NMV']))
    # cursor.executemany('replace FM_mv_tr (SecuCode, TradingDay, TurnoverRate, NMV) \
    #                    value (%s, %s, %s, %s)', values)
    # conn2.commit()

    return mv_tr


def get_historical_mv_tr(stock_list, start):
    historical_mv_trs = list()
    for code in stock_list:
        query = """
        SELECT
            SecuCode, TradingDay, TurnoverRate, NMV
        FROM FM_mv_tr
            WHERE SecuCode = '%s'
            AND TradingDay <= '%s'
            ORDER BY TradingDay desc
            limit 251
            """ % (code, start)
        historical_mv_tr = pd.read_sql(query, DbUtil.get_conn('zj_data'))
        historical_mv_trs.append(historical_mv_tr)
    historical_mv_trs = pd.concat(historical_mv_trs, ignore_index=True, sort=False)
    return historical_mv_trs


def generate_liquidity_descriptors(df, start, end):
    df['stom'] = df.groupby('SecuCode')['TurnoverRate'].\
        apply(lambda x: np.log(x.rolling(window=21, min_periods=21).sum()))
    df['stoq'] = df.groupby('SecuCode')['TurnoverRate'].\
        apply(lambda x: np.log(x.rolling(window=63, min_periods=63).sum()/3))
    df['stoa'] = df.groupby('SecuCode')['TurnoverRate'].\
        apply(lambda x: np.log(x.rolling(window=252, min_periods=252).sum()/12))
    df = df[['SecuCode', 'TradingDay', 'stom', 'stoq', 'stoa']]
    df = df[(df['TradingDay'] > start) & (df['TradingDay'] <= end)]
    return df


def generate_size(df):
    df['size'] = np.log(df['NMV'])
    df = df[['SecuCode', 'TradingDay', 'size', 'NMV']]
    return df


def get_fundamentals(stock_list, end):
    fundamentals = list()
    for code in stock_list:
        query = """
        SELECT
            SM.SecuCode, LCMIN.EndDate, LCMIN.EPS EPS_Raw, LCMIN.NetAssetPS NAPS,
            LCMIN.BasicEPSYOY growth, LCMIN.DebtAssetsRatio leverage
        FROM LC_MainIndexNew LCMIN
            JOIN SecuMain SM ON LCMIN.CompanyCode = SM.CompanyCode
            WHERE SM.SecuCode = '%s'
            AND SM.SecuCategory = 1
            AND LCMIN.EndDate <= '%s'
            ORDER BY EndDate desc
            limit 12
            """ % (code, end)
        fundamental = pd.read_sql(query, DbUtil.get_conn('jydb'))
        fundamentals.append(fundamental)
    fundamentals = pd.concat(fundamentals, ignore_index=True, sort=False)
    fundamentals['EndDate'] = pd.to_datetime(fundamentals['EndDate'], errors='coerce')
    fundamentals['year'] = fundamentals['EndDate'].dt.year
    fundamentals['quarter'] = fundamentals['EndDate'].dt.quarter

    fundamentals = fundamentals.sort_values(by=['SecuCode', 'year', 'quarter'])
    fundamentals_pivot = fundamentals.pivot_table(values='EPS_Raw', columns='SecuCode', index=['year', 'quarter'])
    fundamentals_pivot = fundamentals_pivot.reset_index()
    EPS_Raw = pd.melt(fundamentals_pivot, id_vars=['year', 'quarter'], value_name='EPS_Raw')
    EPS_Temp = EPS_Raw.groupby(['SecuCode', 'year'])['EPS_Raw'].apply(lambda x: x - x.shift(1))
    EPS_Raw['EPS'] = EPS_Temp.reset_index(level=0, drop=True)
    EPS_Raw.loc[EPS_Raw['quarter'] == 1, 'EPS'] = EPS_Raw['EPS_Raw']
    EPS_Raw = EPS_Raw.drop('EPS_Raw', axis=1)
    fundamentals = pd.merge(fundamentals, EPS_Raw, on=['SecuCode', 'year', 'quarter'], how='left')
    fundamentals = fundamentals.drop('EPS_Raw', axis=1)

    # When doing cross-sectional regression, log return is regressed on last quarter's fundamentals
    fundamentals.loc[fundamentals['quarter'] == 4, 'year'] = fundamentals['year'] + 1
    fundamentals['quarter_temp'] = fundamentals['quarter'] + 1
    fundamentals.loc[fundamentals['quarter_temp'] == 5, 'quarter_temp'] = 1
    fundamentals = fundamentals.drop('quarter', axis=1)
    fundamentals = fundamentals.rename(columns={'quarter_temp': 'quarter'})
    return fundamentals


def fill_last_quarter_fundamentals(df, end):

    enddate_quarter_end = pd.to_datetime(end) + pd.offsets.QuarterEnd()
    enddate_fund_available = pd.to_datetime(max(df['EndDate']) + dt.timedelta(days=1)) + pd.offsets.QuarterEnd()
    if enddate_quarter_end != enddate_fund_available:
        add = pd.DataFrame({'SecuCode': ['000001'], 'EndDate': [enddate_quarter_end]})
        add['NAPS'] = np.nan
        add['growth'] = np.nan
        add['leverage'] = np.nan
        add['year'] = add['EndDate'].dt.year
        add['EPS'] = np.nan
        add['quarter'] = add['EndDate'].dt.quarter
        df = df.append(add)
    fundamental_list = ['growth', 'leverage', 'EPS', 'NAPS']
    for fund in fundamental_list:
        df[fund] = pd.to_numeric(df[fund], errors='coerce')
    df_fill = pd.DataFrame()
    for i in range(len(fundamental_list)):
        fund = fundamental_list[i]
        df_pivot = df.pivot_table(values=fund, columns='SecuCode', index=['year', 'quarter'], dropna=False)
        df_pivot = df_pivot.reset_index()
        df_pivot = df_pivot.fillna(method='ffill')
        melt = pd.melt(df_pivot, id_vars=['year', 'quarter'], value_name=fund)
        if i == 0:
            df_fill = melt
        else:
            df_fill = pd.merge(df_fill, melt, on=['year', 'quarter', 'SecuCode'], how='outer')
    return df_fill


def get_industry(stock_list):
    query = """
    SELECT
        SM.SecuCode, LCEI.FirstIndustryName Industry
    FROM SecuMain SM
        JOIN LC_ExgIndustry LCEI ON SM.CompanyCode = LCEI.CompanyCode
        WHERE SM.SecuCode in %s
        AND SM.SecuCategory = 1
        AND LCEI.Standard = 24
        AND LCEI.CancelDate is NULL
        """ % (str(tuple(stock_list)))
    industries = pd.read_sql(query, DbUtil.get_conn('jydb'))
    industry_code = pd.DataFrame({'Industry': ['银行',
                                               '房地产',
                                               '医药生物',
                                               '公用事业',
                                               '综合',
                                               '机械设备',
                                               '建筑装饰',
                                               '建筑材料',
                                               '家用电器',
                                               '汽车',
                                               '食品饮料',
                                               '电子',
                                               '计算机',
                                               '交通运输',
                                               '轻工制造',
                                               '通信',
                                               '休闲服务',
                                               '传媒',
                                               '农林牧渔',
                                               '商业贸易',
                                               '化工',
                                               '有色金属',
                                               '非银金融',
                                               '电气设备',
                                               '国防军工',
                                               '采掘',
                                               '纺织服装',
                                               '钢铁'],
                                   'industry_code': np.arange(28)})
    industries = pd.merge(industries, industry_code, on=['Industry'], how='left')
    return industries


def daily_return(df):
    df = df.sort_values(by=['SecuCode', 'TradingDay'])
    df['log_ret'] = df.groupby('SecuCode')['Price'].apply(lambda x: np.log(x) - np.log(x).shift(1))
    return df


def merge_industry(df1, df2):
    df = pd.merge(df1, df2[['SecuCode', 'Industry', 'industry_code']],
                  on=['SecuCode'], how='inner')
    # The stocks with unknown industry are dropped from the whole estimation universe.
    # df = df.dropna(subset = ['Industry'])
    return df


def merge_fundamentals(df1, df2):
    # If a stock has no fundamental record at all for a specific quarter, it is dropped
    # from the estimation universe of that quarter.
    df = pd.merge(df1, df2, on=['SecuCode', 'year', 'quarter'], how='inner')
    return df


def merge_mv(df1, df2):
    # By right join, stock data during suspension is dropped from our estimation universe.
    df = pd.merge(df1, df2, on=['SecuCode', 'TradingDay'], how='inner')
    return df


def generate_momentum(df, start, end):
    df_new = df.sort_values(by=['SecuCode', 'TradingDay'])
    temp = df_new.groupby('SecuCode')['log_ret'].rolling(504).\
        apply(lambda x: pd.Series(x).ewm(halflife=126).mean().iloc[-1])
    df_new['momentum'] = temp.reset_index(level=0, drop=True)
    df_new['date'] = df_new.groupby('SecuCode', as_index=False)['TradingDay'].shift(-20)
    df_new = df_new.drop('TradingDay', axis=1)
    df_new = df_new.rename(columns={'date': 'TradingDay'})
    df_new = df_new[['SecuCode', 'TradingDay', 'momentum']]
    df_new = df_new[(df_new['TradingDay'] > start) & (df_new['TradingDay'] <= end)]
    return df_new


def generate_dastd(df, start, end, ndays):
    df = df.sort_values(by = ['SecuCode', 'TradingDay'])
    df_new = df.groupby('SecuCode').apply(lambda x: x.sort_values('TradingDay').iloc[-252 - ndays:])
    df_new = df_new.reset_index(level=0, drop=True)
    temp = df_new.groupby('SecuCode')['log_ret'].rolling(252).apply(lambda x: pd.Series(x).ewm(halflife=42).std().iloc[-1])
    df_new['dastd'] = temp.reset_index(level=0, drop=True)
    df_new = df_new[['SecuCode', 'TradingDay', 'dastd']]
    df_new = df_new[(df_new['TradingDay'] > start) & (df_new['TradingDay'] <= end)]
    return df_new


def cumulative_range(sr):
    zs = [sr[-x * 21:].sum() for x in list(range(1, 13))]
    max_z = np.max(zs)
    min_z = np.min(zs)
    cmra = max_z - min_z
    return cmra


def generate_cmra(df, start, end, ndays):
    df = df.sort_values(by=['SecuCode', 'TradingDay'])
    df_new = df.groupby('SecuCode').apply(lambda x: x.sort_values('TradingDay').iloc[-252 - ndays:])
    df_new = df_new.reset_index(level=0, drop=True)
    cmra = df_new.groupby('SecuCode')['log_ret'].rolling(252).apply(cumulative_range)
    df_new['cmra'] = cmra.reset_index(level=0, drop=True)
    df_new = df_new[['SecuCode', 'TradingDay', 'cmra']]
    df_new = df_new[(df_new['TradingDay'] > start) & (df_new['TradingDay'] <= end)]
    return df_new


def generate_btop_etop(df):
    df['book_to_price'] = df['NAPS']/df['Price']
    df['earnings_yield'] = df['EPS']/df['Price']
    return df


def get_critical_date(df):
    code_list = df.SecuCode.unique().tolist()
    query = """
    select SecuCode, ListedDate
    from SecuMain
        where SecuCode in %s
        and SecuCategory = 1
        and SecuMarket in (83, 90)
    """ % (str(tuple(code_list)))
    listed_dates = pd.read_sql(query, DbUtil.get_conn('jydb'))
    listed_dates['CriticalDates'] = listed_dates['ListedDate'] + pd.DateOffset(months = 12)
    critical_dates = listed_dates.drop('ListedDate', axis=1)
    return critical_dates


def generate_cap_weight(df):
    df['cap_weight'] = df.groupby('TradingDay')['NMV'].apply(lambda x: x/x.sum())
    return df


def generate_beta_residual(df):
    df = df.sort_values(by=['SecuCode', 'TradingDay'])
    df = df.dropna(subset=['log_ret'])

    stocks = df.SecuCode.unique()
    market_ret = df.groupby('TradingDay').apply(lambda x: (x['cap_weight'] * x['log_ret']).sum())
    market_ret = market_ret.reset_index()
    market_ret = market_ret.rename(columns={0: 'market_ret'})
    df = pd.merge(df, market_ret, on=['TradingDay'], how='left')
    index = np.arange(0, 252)
    w = [np.exp(np.log(0.5)/63)**(251 - i) for i in index]
    j = 0
    Betas = pd.DataFrame()
    for stock in stocks:
        j = j + 1
        stock_data = df[df['SecuCode'] == stock][['TradingDay', 'log_ret', 'market_ret']]
        nrows = stock_data.shape[0]
        if nrows >= 252:
            for i in range(0, nrows - 251):
                x = stock_data['market_ret'].iloc[i:i + 252]
                x = sm.add_constant(x)
                y = stock_data['log_ret'].iloc[i:i + 252]
                models = sm.WLS(y, x, w).fit()
                beta = models.params['market_ret']
                resid = models.resid
                resid_vol = resid.std()
                TradingDay = stock_data.iloc[i + 251].TradingDay
                Beta = pd.DataFrame({'SecuCode': [stock], 'TradingDay': [TradingDay],
                                     'beta': [beta], 'resid_vol': [resid_vol]})
                Betas = Betas.append(Beta)
    Betas = Betas[['SecuCode', 'TradingDay', 'beta', 'resid_vol']]
    logger.info('length of Betas: %s' % (len(Betas)))
    DbHandleUtil.save('FM_Betas', Betas, 'zj_data', has_jsid=False)
    # values = list(zip(Betas['SecuCode'], Betas['TradingDay'], Betas['beta'], Betas['resid_vol']))
    # cursor.executemany('replace FM_Betas (SecuCode, TradingDay, beta, resid_vol) \
    #                     value (%s, %s, %s, %s)', values)
    # conn.commit()

    return Betas

def generate_non_linear_size(df):
    df['size_cubed'] = np.power(df['size'], 3)
    temp = df.groupby('TradingDay')['sqrtmarketcap'].apply(lambda x: x/x.sum())
    df['regression_weight'] = temp.reset_index(level=0, drop=True)
    b = df.groupby('TradingDay').apply(lambda x: (x['regression_weight'] * x['size_cubed'] * x['size']).sum()/
                                                 ((x['regression_weight'] * x['size'] * x['size']).sum()))
    b = b.reset_index()
    b = b.rename(columns={0: 'b'})
    df = pd.merge(df, b, on='TradingDay', how='left')
    df['non_linear_size'] = df['size_cubed'] - df['size'] * df['b']

    return df


#%% Fill missing values
def fill_miss(df, varlist, by1, by2, by3):
    for ff in varlist:
        df[ff][(df[ff] == float('inf'))|(df[ff] == float('-inf'))] = np.nan
    df[varlist] = df.groupby(by1, as_index=False)[varlist].transform(lambda x: x.fillna(x.median()))
    if len(by2) > 0:
            df[varlist] = df.groupby(by2, as_index=False)[varlist].transform(lambda x: x.fillna(x.median()))
            if len(by3) > 0:
                 df[varlist] = df.groupby(by3, as_index=False)[varlist].transform(lambda x: x.fillna(x.median()))
    return df

#%% Outlier cleaning
def standardize_series(s, num):
    mean, std = np.nanmean(s), np.nanstd(s)
    r_outliers = s > mean + num * std
    l_outliers = s < mean - num * std
    s[r_outliers] = mean + num * std
    s[l_outliers] = mean - num * std
    return s


def outlier_method(df, varlist, by=[]):
    num = 3
    if len(by) > 0:
        for ff in varlist:
            df[ff] = df.groupby(by)[ff].apply(standardize_series, num=num)
    elif len(by) == 0:
        for ff in varlist:
            df[ff] = standardize_series(df[ff], num=num)
    return df


def normalize_series(df, ff, weight):
    std = np.nanstd(df[ff])
    df[ff] = (df[ff] - (df[ff] * df[weight]).sum())/std
    return df


def data_preprocess(df, varlist, outlier_by=[], normalize_by=['TradingDay']):
    for ff in varlist:
        df[ff][df[ff] == np.inf] = np.nan
    df = outlier_method(df, varlist, by=[])
    for ff in varlist:
       df = df.groupby(normalize_by).apply(lambda x: normalize_series(x, ff=ff, weight='cap_weight'))

    return df


#%% Residual vol is orthogonalized with respect to liquidity and beta
#   liquidity is orthogonalized with respect to size, beta, residual vol
def orthogonalize_residual_vol(df):
    temp = df.groupby('TradingDay', squeeze=True).apply(lambda x: x['residual_vol'] - np.dot(x['residual_vol'],
                                                                    x['beta'])/np.dot(x['beta'], x['beta']) * x['beta'])
    df['orthogonalized_residual_vol'] = temp.reset_index(level=0, drop=True)
    return df


def orthogonalize_liquidity(df):
    temp = df.groupby('TradingDay', squeeze=True).apply(lambda x: x['liquidity'] - np.dot(x['liquidity'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']
                                          - np.dot(x['liquidity'], (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']))/
                                          np.dot((x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']),
                                                 (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta'])) *
                                                 (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta'])
                                          - np.dot(x['liquidity'], (x['size'] - np.dot(x['size'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']
                                          - np.dot(x['size'], (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']))/
                                          np.dot((x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']),
                                                 (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta'])) *
                                                 (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta'])))/
                                                 np.dot((x['size'] - np.dot(x['size'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']
                                          - np.dot(x['size'], (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']))/
                                          np.dot((x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']),
                                                 (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta'])) *
                                                 (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta'])), (x['size'] - np.dot(x['size'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']
                                          - np.dot(x['size'], (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']))/
                                          np.dot((x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']),
                                                 (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta'])) *
                                                 (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']))) * (x['size'] - np.dot(x['size'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']
                                          - np.dot(x['size'], (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']))/
                                          np.dot((x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta']),
                                                 (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta'])) *
                                                 (x['residual_vol'] - np.dot(x['residual_vol'], x['beta'])/np.dot(x['beta'], x['beta']) * x['beta'])))
    df['orthogonalized_liquidity'] = temp.reset_index(level=0, drop=True)
    return df

#%% Generate industry factors
def industry_factors(df):
    df['industry_code'] = df['industry_code'].astype(str)
    td = pd.get_dummies(df['industry_code'], dummy_na=False, prefix='ind', drop_first=False)
    df = df.join(td)
    return df


def implied_factor_return(df_all, df, y, x, ind_list, factor_list, factor_num):
    df = df.sort_values(by='TradingDay')
    df_all = df_all.sort_values(by='TradingDay')
    beta = pd.DataFrame()
    constraint = pd.DataFrame()
    r2 = pd.DataFrame()
    adjr2 = pd.DataFrame()
    resid = pd.DataFrame()
    resid_all = pd.DataFrame()
    yhat = pd.DataFrame()
    yhat_all = pd.DataFrame()
    tscore = pd.DataFrame()
    for t in df.TradingDay.unique():
        print(t)
        data_t = df.loc[df.TradingDay == t, :]
        data_t_all = df_all.loc[df_all.TradingDay == t, :]
        r = np.matrix(data_t[y]).T
        X = np.matrix(data_t[x])
        r_all = np.matrix(data_t_all[y]).T
        X_all = np.matrix(data_t_all[x])
        stock_list = data_t['SecuCode'].unique().tolist()
        stock_num = len(stock_list)
        regression_weight = data_t['sqrtmarketcap']/(data_t['sqrtmarketcap'].sum())
        W = np.matrix(np.diag(regression_weight))
        q1 = np.repeat(0, 11)
        q2 = []
        for ind in ind_list:
            ind_w = data_t[data_t[ind] == 1]['marketcap'].sum()/(data_t['marketcap'].sum())
            q2.append(ind_w)

        q = np.concatenate((q1, np.asarray(q2)), axis=0)
        q = np.matrix(q)
        pin = np.linalg.pinv(2 * X.T * W * X)
        f = (pin - pin * q.T * np.linalg.pinv(q * pin * q.T) * q * pin) * 2 * X.T * W * r
        params = pd.DataFrame(f.T, columns=factor_list)
        params['TradingDay'] = t
        cons = q * f
        cons = pd.DataFrame(cons)
        cons['TradingDay'] = t
        cons = cons.rename(columns={0: 'ind_cap_sum'})

        res = r - X * f
        resid_sq = np.multiply((r - X * f), (r - X * f))
        res = np.append(res, resid_sq, axis=1)
        res = pd.DataFrame(res, columns=['wls_resid', 'wls_resid2'])
        res = data_t[['TradingDay', 'SecuCode']].reset_index().drop(columns=['index'], axis=1).\
              merge(res, left_index=True, right_index=True)

        tot = r - np.matrix(regression_weight) * r
        r_sq = 1 - np.matrix(regression_weight) * resid_sq/(np.matrix(regression_weight) * np.multiply(tot, tot))
        r_sq = pd.DataFrame(r_sq, columns=['r2'])
        r_sq['TradingDay'] = t

        r_adj = 1 - (1 - r_sq.r2) * (stock_num - 1)/(stock_num - 1 - factor_num)
        r_adj = pd.DataFrame(r_adj)
        r_adj = r_adj.rename(columns={'r2': 'adjr2'})
        r_adj['TradingDay'] = t

        yt = pd.DataFrame(X * f, columns=['wls_yhat'])
        yt = data_t[['TradingDay', 'SecuCode']].reset_index().drop(columns=['index'], axis=1).\
             merge(yt, left_index=True, right_index=True)

        yt_all = pd.DataFrame(X_all * f, columns=['wls_yhat'])
        yt_all = data_t_all[['TradingDay', 'SecuCode']].reset_index().drop(columns=['index'], axis=1).\
             merge(yt_all, left_index=True, right_index=True)

        tvalue = np.divide(f.T, np.sqrt(np.divide(resid_sq.sum()/(stock_num - 38), np.power(X - X.mean(0), 2).sum(0))))
        tvalue = pd.DataFrame(tvalue, columns=factor_list)
        tvalue['TradingDay'] = t
        tvalue = tvalue.drop('country', axis=1)

        res_all = r_all - X_all * f
        resid_sq_all = np.multiply((r_all - X_all * f), (r_all - X_all * f))
        res_all = np.append(res_all, resid_sq_all, axis=1)
        res_all = pd.DataFrame(res_all, columns=['wls_resid', 'wls_resid2'])
        res_all = data_t_all[['TradingDay', 'SecuCode']].reset_index().drop(columns=['index'], axis=1).\
              merge(res_all, left_index=True, right_index=True)

        beta = beta.append(params)
        constraint = constraint.append(cons)
        resid = resid.append(res)
        resid_all = resid_all.append(res_all)
        r2 = r2.append(r_sq)
        adjr2 = adjr2.append(r_adj)
        yhat = yhat.append(yt)
        yhat_all = yhat_all.append(yt_all)
        tscore = tscore.append(tvalue)

    return beta, constraint, resid, resid_all, r2, adjr2, yhat, yhat_all, tscore

#%% Upload for Optimization
def get_listed_secucodes():
    query = """
    SELECT SecuCode
    FROM SecuMain
    WHERE SecuCategory = 1
    and ListedState = 1
    and SecuMarket in (83, 90)
    """
    secucodes = pd.read_sql(query, DbUtil.get_conn('jydb'))
    listed_codes = secucodes.SecuCode.tolist()
    return listed_codes


def model_ret_fill(df, factor_list):
    df = df.sort_values(by=['SecuCode', 'TradingDay'])
    model_ret_fill = pd.DataFrame()
    for i in range(len(factor_list)):
        factor = factor_list[i]
        pivot = df.pivot_table(values=factor, columns='SecuCode', index='TradingDay')
        pivot = pivot.fillna(method='ffill')
        pivot = pivot.reset_index()
        melt = pd.melt(pivot, id_vars='TradingDay', value_name=factor)
        if i == 0:
            model_ret_fill = melt
        else:
            model_ret_fill = pd.merge(model_ret_fill, melt, on=['SecuCode', 'TradingDay'], how='left')
    model_ret_filled = model_ret_fill.dropna(subset=['beta'])
    return model_ret_filled


def factor_return_vol_forecast(tradingdays, factor_list):
    query = """
    SELECT TradingDay, country, beta, momentum, size, earnings_yield, residual_vol, growth, book_to_price, leverage,
    liquidity, non_linear_size, ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, ind_10, ind_11,
    ind_12, ind_13, ind_14, ind_15, ind_16, ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, ind_25,
    ind_26, ind_27
    FROM FM_WLS_Beta
    """
    wls_beta = pd.read_sql(query, DbUtil.get_conn('zj_data'))
    wls_beta = wls_beta.sort_values(by='TradingDay')
    PreBetaVol = pd.DataFrame()
    PreBetaVol_5d = pd.DataFrame()
    PreBetaVol_5p = pd.DataFrame()
    forecasted_factor_return = pd.DataFrame()

    for tradingday in tradingdays:
        wls_beta_used = wls_beta[wls_beta['TradingDay'] <= tradingday]
        beta_conditional_vol_1d = pd.DataFrame()
        beta_conditional_vol_5d = pd.DataFrame()
        beta_conditional_vol_5p = pd.DataFrame()
        e_wls_beta_all = pd.DataFrame()

        for ff in factor_list:
            am = arch_model(y=1000 * wls_beta_used[ff].values, mean='ARX', lags=2, vol='GARCH',
                            p=1, o=0, q=1, power=2.0, dist='Normal', hold_back=None)
            res = am.fit(update_freq=5, disp='off')
            forecasts = res.forecast(horizon=5)
            beta_conditional_vol1 = np.sqrt(forecasts.variance.dropna()).transpose().values[0]/1000
            beta_conditional_vol5 = forecasts.variance.dropna().transpose().values/1000000
            beta_conditional_vol_1d[ff] = pd.DataFrame(beta_conditional_vol1, columns=[ff])
            beta_conditional_vol_5d[ff] = pd.DataFrame({ff: [np.sqrt(beta_conditional_vol5.sum())]})
            beta_conditional_vol_5p[ff] = np.concatenate(np.sqrt(beta_conditional_vol5))
            ar_params = res.params.iloc[0:3]
            ar_pvalues = res.pvalues.iloc[0:3]
            ar_params.loc[ar_params.index.isin(ar_pvalues[ar_pvalues > 0.1].index)] = 0
            wls_beta_used_lags = 1000 * wls_beta_used[ff].iloc[-2:]
            e_wls_beta = ar_params.Const + ar_params[ar_params.index[1]] * wls_beta_used_lags.iloc[1] + \
                         ar_params[ar_params.index[2]] * wls_beta_used_lags.iloc[0]
            e_wls_beta_all[ff] = pd.DataFrame([e_wls_beta/1000], columns=[ff])
        beta_conditional_vol_1d['TradingDay'] = tradingday
        beta_conditional_vol_5d['TradingDay'] = tradingday
        beta_conditional_vol_5p['TradingDay'] = tradingday
        beta_conditional_vol_5p['forecast_period'] = np.arange(5) + 1
        e_wls_beta_all['TradingDay'] = tradingday
        PreBetaVol = PreBetaVol.append(beta_conditional_vol_1d)
        PreBetaVol_5d = PreBetaVol_5d.append(beta_conditional_vol_5d)
        PreBetaVol_5p = PreBetaVol_5p.append(beta_conditional_vol_5p)
        forecasted_factor_return = forecasted_factor_return.append(e_wls_beta_all)
    return PreBetaVol, PreBetaVol_5d, PreBetaVol_5p, forecasted_factor_return


def factor_return_correlation(tradingdays):
    query = """
    SELECT TradingDay, country, beta, momentum, size, earnings_yield, residual_vol, growth, book_to_price, leverage,
    liquidity, non_linear_size, ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, ind_10, ind_11,
    ind_12, ind_13, ind_14, ind_15, ind_16, ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, ind_25,
    ind_26, ind_27
    FROM FM_WLS_Beta
    """
    wls_beta = pd.read_sql(query, DbUtil.get_conn('zj_data'))
    wls_beta = wls_beta.sort_values(by='TradingDay')
    factor_correlation = pd.DataFrame()
    for tradingday in tradingdays:
        wls_beta_used = wls_beta[wls_beta['TradingDay'] <= tradingday]
        wls_beta_used = wls_beta_used.sort_values(by='TradingDay')
        factor_correlation_by_date = wls_beta_used.corr()
        factor_correlation_by_date = factor_correlation_by_date.reset_index()
        factor_correlation_by_date = factor_correlation_by_date.rename(columns={'index': 'factor'})
        factor_correlation_by_date['TradingDay'] = tradingday
        factor_correlation = factor_correlation.append(factor_correlation_by_date)
    return factor_correlation


def factor_return_correlation_ewma(tradingdays):
    query = """
    SELECT TradingDay, country, beta, momentum, size, earnings_yield, residual_vol, growth, book_to_price, leverage,
    liquidity, non_linear_size, ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, ind_10, ind_11,
    ind_12, ind_13, ind_14, ind_15, ind_16, ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, ind_25,
    ind_26, ind_27
    FROM FM_WLS_Beta
    """
    wls_beta = pd.read_sql(query, DbUtil.get_conn('zj_data'))
    wls_beta = wls_beta.sort_values(by='TradingDay')
    factor_correlation = pd.DataFrame()
    for tradingday in tradingdays:
        wls_beta_used = wls_beta[wls_beta['TradingDay'] <= tradingday]
        wls_beta_used = wls_beta_used.sort_values(by='TradingDay')
        factor_correlation_by_date = wls_beta_used.iloc[:, 1:].ewm(halflife=504).corr()[-39:].\
            reset_index(level=0, drop=True)
        factor_correlation_by_date = factor_correlation_by_date.reset_index()
        factor_correlation_by_date = factor_correlation_by_date.rename(columns={'index': 'factor'})
        factor_correlation_by_date['TradingDay'] = tradingday
        factor_correlation = factor_correlation.append(factor_correlation_by_date)
    return factor_correlation


def iftradingday(start, end):
    query = """
    SELECT TradingDate
    FROM QT_TradingDayNew
        WHERE TradingDate > '%s' and TradingDate <= '%s'
        AND IfTradingDay = 1
        AND SecuMarket = 83
    """ % (start, end)
    tradingdays = pd.read_sql(query, DbUtil.get_conn('jydb'))
    return tradingdays


def is_data_sync_successfully(day, conn):
    next_day = day + dt.timedelta(days=1)
    query = """
        SELECT COUNT(id)
        FROM JYDB.ZJ_Tushare_Autoupdate_info
        WHERE
           Time_start > %s
           AND Time_start < %s
           AND Error_messgae != '{}';
    """ % (day, next_day)
    result = pd.read_sql(query, conn)
    return result.iloc[0][0] == 0


def save_data():

    # get the datdabase connections for the furthur operations
    # jy_conn, zj_conn, fd_conn = construct_db_connections()
    #
    # jy_conn.autocommit = True
    # zj_conn.autocommit = True
    # fd_conn.autocommit = True
    # jy_cursor = jy_conn.cursor()
    # zj_cursor = zj_conn.cursor()
    # fd_cursor = fd_conn.cursor()

    # if not is_data_sync_successfully(dt.date.today(), jy_conn):
    #     return

    start = get_last_run_date()
    end = get_available_date()
    # start = pd.to_datetime('2022-06-27')
    # start = pd.to_datetime('2021-12-31')
    # end = pd.to_datetime('2022-02-18')
    logger.info('start date: %s' % start)
    logger.info('end date: %s' % end)
#    print(end)

    if start < end:

        stock_price = get_stock_price(start=start, end=end)
        tradingdays = iftradingday(start, end)
        tradingdays = tradingdays.TradingDate.tolist()
        ndays = len(tradingdays)
        logger.info('number of tradingdays: %s' % ndays)
        # quarters = stock_price[['year', 'quarter']].drop_duplicates()
        stocks = stock_price.SecuCode.unique()
        logger.info('number of stocks: %s' % len(stocks))
        # Extract 525-days stock price until last run date
        logger.info('start to extract stock price')
        historical_stocks_price = get_historical_stock_price(start=start, stock_list=stocks)
        # Combine stock price information
        stock_price_all = pd.concat([stock_price, historical_stocks_price])
        stock_price_all = stock_price_all.sort_values(by=['SecuCode', 'TradingDay'])
        logger.info('start to extract negotiable value')
        # Extract market value and turnover rate information from last run date up till now
        mv_tr = get_mv_tr(stock_list=stocks, start=start, end=end)
        # Extract 252-days market value and turnover rate information until last run date
        a = time.time()
        historical_mv_trs = get_historical_mv_tr(stock_list=stocks, start=start)
        b = time.time()
        logger.info(b - a)
        # Combine market value and turnover rate information
        mv_tr = pd.concat([mv_tr, historical_mv_trs])
        mv_tr = mv_tr.sort_values(by=['SecuCode', 'TradingDay'])
        logger.info('start to calculate descriptors')
        # Generate liquidity descriptors
        liquidity_descriptors = generate_liquidity_descriptors(mv_tr, start, end)

        # Calculate size
        size = generate_size(mv_tr)
        logger.info('length of size: %s' % len(size))
        logger.info('length of 600519 size: %s' % len(size[size['SecuCode']=='600519']))
        # Read in fundamental indicators
        fundamentals = get_fundamentals(stock_list=stocks, end=end)
        fundamentals = fill_last_quarter_fundamentals(fundamentals, end)

        # Read in industry information, and create industry code
        industries = get_industry(stock_list=stocks)
        logger.info('start to calculate stock returns')
        # Calculate daily log return, and truncate log return to 0.1 and -0.1
        model_ret = daily_return(stock_price_all)
        logger.info('length of model_ret: %s' % len(model_ret))
        logger.info('length of 600519 return: %s' % len(model_ret[model_ret['SecuCode']=='600519']))
        # Merge return data with industry information, and drop stock return data with no industry information
        model_ret = merge_industry(model_ret, industries)

        # Calculate momentum
        momentum = generate_momentum(model_ret, start, end)

        # Calculate residual_volatility descriptor "dastd"
        dastd = generate_dastd(model_ret, start, end, ndays)

        # Calculate residual_volatility descriptor "cumulative range"
        cmra = generate_cmra(model_ret, start, end, ndays)

        # Merge fundamentals with industries
        fundamentals = merge_industry(fundamentals, industries)

        # Fill fundamental indicators' missing data
        fundamentals = fill_miss(fundamentals, ['EPS', 'NAPS', 'growth', 'leverage'],
                              'SecuCode', ['year', 'quarter', 'industry_code'], ['industry_code'])
        fundamentals = fundamentals.drop(['Industry', 'industry_code'], axis=1)

        # Merge return data with fundamental indicators

        model_ret = merge_fundamentals(model_ret, fundamentals)
        logger.info('length of 600519 after fundamentals: %s' % (len(model_ret[model_ret['SecuCode'] == '600519'])))
        logger.info('length of model_ret after fundamentals: %s' % len(model_ret))
        # Merge return data with market value data
        model_ret = merge_mv(model_ret, size)
        logger.info('length of model_ret after size: %s' % len(model_ret))
        logger.info('length of 600519 after size: %s' % (len(model_ret[model_ret['SecuCode'] == '600519'])))
        # Calculate BTOP and ETOP
        model_ret = generate_btop_etop(model_ret)

        # Calculate cap_weight
        model_ret = generate_cap_weight(model_ret)

        # Calculate Betas and residuals
        logger.info('length of model_ret before beta: %s' % len(model_ret))
        Betas = generate_beta_residual(model_ret[['SecuCode', 'TradingDay', 'log_ret', 'cap_weight']])

        # Merge Betas and residual_vol to return data
        model_ret = model_ret[(model_ret['TradingDay'] > start) & (model_ret['TradingDay'] <= end)]
        model_ret = pd.merge(model_ret, Betas, on=['SecuCode', 'TradingDay'], how='left')

        # Merge other descriptors and factors to return data
        model_ret = pd.merge(model_ret, liquidity_descriptors, on=['SecuCode', 'TradingDay'], how='left')
        model_ret = pd.merge(model_ret, momentum, on=['SecuCode', 'TradingDay'], how='left')
        model_ret = pd.merge(model_ret, dastd, on=['SecuCode', 'TradingDay'], how='left')
        model_ret = pd.merge(model_ret, cmra, on=['SecuCode', 'TradingDay'], how='left')

        # Fill other descriptors and factors' missing data
        model_ret = fill_miss(model_ret, ['stom', 'stoq', 'stoa', 'momentum', 'dastd', 'cmra', 'beta', 'resid_vol'],
                             'SecuCode', ['TradingDay', 'industry_code'], [])
        logger.info('start to preprocess descriptors')
        # Preprocess descriptors
        model_ret = data_preprocess(model_ret, ['stom', 'stoq', 'stoa', 'dastd', 'cmra', 'resid_vol'])

        # Create Liquidity and Residual Volatility
        model_ret['liquidity'] = model_ret['stom'] * 0.35 + model_ret['stoq'] * 0.35 + model_ret['stoa'] * 0.3
        model_ret['residual_vol'] = model_ret['dastd'] * 0.74 + model_ret['cmra'] * 0.16 + model_ret['resid_vol'] * 0.1

        # Preprocess factors
        factor_list = ['beta', 'momentum', 'size', 'earnings_yield', 'residual_vol', 'growth',
                    'book_to_price', 'leverage', 'liquidity']

        model_ret = data_preprocess(model_ret, factor_list)

        # Create country factor
        model_ret['country'] = 1

        model_ret = model_ret.rename(columns={'NMV': 'marketcap'})
        model_ret['sqrtmarketcap'] = np.sqrt(model_ret['marketcap'])

        # Create non-linear size factor
        model_ret = generate_non_linear_size(model_ret)
        model_ret = data_preprocess(model_ret, ['non_linear_size'])

        # Orthogonalize liquidity and residual vol
        model_ret = orthogonalize_residual_vol(model_ret)
        model_ret = orthogonalize_liquidity(model_ret)
        model_ret = data_preprocess(model_ret, ['orthogonalized_residual_vol', 'orthogonalized_liquidity'])
        model_ret = model_ret.drop(['residual_vol', 'liquidity'], axis=1)
        model_ret = model_ret.rename(columns={'orthogonalized_residual_vol': 'residual_vol',
                                              'orthogonalized_liquidity': 'liquidity'})

        # Create industry factors
        model_ret = industry_factors(model_ret)
        ind_list = ['ind_' + str(x) for x in range(0, 28)]
        keep = ['SecuCode', 'TradingDay', 'country', 'beta',
                'momentum', 'size', 'earnings_yield', 'residual_vol', 'growth',
                'book_to_price', 'leverage', 'liquidity', 'non_linear_size'] + ind_list
        factor_exposure_by_date = model_ret[keep]
        listed_stocks = get_listed_secucodes()
        query = """
        SELECT SecuCode, TradingDay, country, beta, momentum, size, earnings_yield, residual_vol, growth, book_to_price, 
        leverage, liquidity, non_linear_size, ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, 
        ind_10, ind_11, ind_12, ind_13, ind_14, ind_15, ind_16, ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, 
        ind_24, ind_25, ind_26, ind_27
        FROM FM_FactorExposure
        WHERE TradingDay = '%s' and SecuCode in %s
        """ % (start, str(tuple(listed_stocks)))
        last_FactorExpo = pd.read_sql(query, DbUtil.get_conn('zj_data'))
        model_ret_for_opt = pd.concat([last_FactorExpo, factor_exposure_by_date])
        factor_list = ['country', 'beta', 'momentum', 'size', 'earnings_yield',
                       'residual_vol', 'growth', 'book_to_price', 'leverage', 'liquidity', 'non_linear_size'] + ind_list
        model_ret_filled = model_ret_fill(model_ret_for_opt, factor_list)
        factor_exposure_filled = model_ret_filled[model_ret_filled['TradingDay'] != start]
        factor_exposure_filled = factor_exposure_filled.replace({pd.NaT: None})
        factor_exposure_filled = factor_exposure_filled.where(pd.notnull(factor_exposure_filled), None)
        factor_list = ['country', 'beta', 'momentum', 'size', 'earnings_yield', 'residual_vol',
                   'growth', 'book_to_price', 'leverage', 'liquidity', 'non_linear_size', 'ind_0',
                   'ind_1', 'ind_2', 'ind_3', 'ind_4', 'ind_5', 'ind_6', 'ind_7', 'ind_8', 'ind_9', 'ind_10',
                   'ind_11', 'ind_12', 'ind_13', 'ind_14', 'ind_15', 'ind_16', 'ind_17', 'ind_18', 'ind_19',
                   'ind_20', 'ind_21', 'ind_22', 'ind_23', 'ind_24', 'ind_25', 'ind_26', 'ind_27']
        columns = ['SecuCode', 'TradingDay'] + factor_list
        factor_exposure_filled = factor_exposure_filled[columns]
        DbHandleUtil.save('FM_FactorExposure', factor_exposure_filled, 'zj_data', has_jsid=False)
        # values = list(zip(factor_exposure_filled['SecuCode'], factor_exposure_filled['TradingDay'],
        #                   factor_exposure_filled['country'], factor_exposure_filled['beta'], factor_exposure_filled['momentum'], factor_exposure_filled['size'],
        #                   factor_exposure_filled['earnings_yield'], factor_exposure_filled['residual_vol'], factor_exposure_filled['growth'],
        #                   factor_exposure_filled['book_to_price'], factor_exposure_filled['leverage'], factor_exposure_filled['liquidity'], factor_exposure_filled['non_linear_size'],
        #                   factor_exposure_filled['ind_0'], factor_exposure_filled['ind_1'], factor_exposure_filled['ind_2'], factor_exposure_filled['ind_3'],
        #                   factor_exposure_filled['ind_4'], factor_exposure_filled['ind_5'], factor_exposure_filled['ind_6'], factor_exposure_filled['ind_7'],
        #                   factor_exposure_filled['ind_8'], factor_exposure_filled['ind_9'], factor_exposure_filled['ind_10'], factor_exposure_filled['ind_11'],
        #                   factor_exposure_filled['ind_12'], factor_exposure_filled['ind_13'], factor_exposure_filled['ind_14'], factor_exposure_filled['ind_15'],
        #                   factor_exposure_filled['ind_16'], factor_exposure_filled['ind_17'], factor_exposure_filled['ind_18'], factor_exposure_filled['ind_19'],
        #                   factor_exposure_filled['ind_20'], factor_exposure_filled['ind_21'], factor_exposure_filled['ind_22'], factor_exposure_filled['ind_23'],
        #                   factor_exposure_filled['ind_24'], factor_exposure_filled['ind_25'], factor_exposure_filled['ind_26'], factor_exposure_filled['ind_27']))
        #
        #
        # zj_cursor.executemany('replace FM_FactorExposure (SecuCode, TradingDay, \
        #                           country, beta, momentum, size, earnings_yield, residual_vol,\
        #                           growth, book_to_price, leverage, liquidity, non_linear_size, ind_0, ind_1, \
        #                           ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, \
        #                           ind_10, ind_11, ind_12, ind_13, ind_14, ind_15, ind_16, \
        #                           ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, \
        #                           ind_25, ind_26, ind_27) value (%s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', values)
        #
        #
        # zj_conn.commit()

        model_ret_for_opt = model_ret_for_opt.sort_values(by=['SecuCode', 'TradingDay'])
        model_ret_for_opt['TradingDay_shift'] = model_ret_for_opt.groupby('SecuCode')['TradingDay'].shift(-1)
        model_ret_for_opt = model_ret_for_opt.drop('TradingDay', axis=1)
        model_ret_for_opt = model_ret_for_opt.rename(columns={'TradingDay_shift': 'TradingDay'})
        model_ret_for_opt = model_ret_for_opt.dropna()
        keep = ['SecuCode', 'TradingDay', 'log_ret', 'sqrtmarketcap', 'marketcap']
        model_ret_for_regression = model_ret[keep]
        model_ret_for_regression = pd.merge(model_ret_for_regression, model_ret_for_opt,
                                            on=['SecuCode', 'TradingDay'], how='left')
        model_ret_for_regression = model_ret_for_regression.dropna()
        columns = ['SecuCode', 'TradingDay', 'log_ret', 'sqrtmarketcap', 'marketcap'] + factor_list
        model_ret_for_regression = model_ret_for_regression[columns]
        DbHandleUtil.save('FM_Cleaned_Data', model_ret_for_regression, 'zj_data', has_jsid=False)
        # values = list(zip(model_ret_for_regression['SecuCode'], model_ret_for_regression['TradingDay'], model_ret_for_regression['log_ret'],
        #                   model_ret_for_regression['sqrtmarketcap'], model_ret_for_regression['marketcap'], model_ret_for_regression['country'],
        #                   model_ret_for_regression['beta'], model_ret_for_regression['momentum'], model_ret_for_regression['size'],
        #                   model_ret_for_regression['earnings_yield'], model_ret_for_regression['residual_vol'], model_ret_for_regression['growth'],
        #                   model_ret_for_regression['book_to_price'], model_ret_for_regression['leverage'], model_ret_for_regression['liquidity'], model_ret_for_regression['non_linear_size'],
        #                   model_ret_for_regression['ind_0'], model_ret_for_regression['ind_1'], model_ret_for_regression['ind_2'], model_ret_for_regression['ind_3'],
        #                   model_ret_for_regression['ind_4'], model_ret_for_regression['ind_5'], model_ret_for_regression['ind_6'], model_ret_for_regression['ind_7'],
        #                   model_ret_for_regression['ind_8'], model_ret_for_regression['ind_9'], model_ret_for_regression['ind_10'], model_ret_for_regression['ind_11'],
        #                   model_ret_for_regression['ind_12'], model_ret_for_regression['ind_13'], model_ret_for_regression['ind_14'], model_ret_for_regression['ind_15'],
        #                   model_ret_for_regression['ind_16'], model_ret_for_regression['ind_17'], model_ret_for_regression['ind_18'], model_ret_for_regression['ind_19'],
        #                   model_ret_for_regression['ind_20'], model_ret_for_regression['ind_21'], model_ret_for_regression['ind_22'], model_ret_for_regression['ind_23'],
        #                   model_ret_for_regression['ind_24'], model_ret_for_regression['ind_25'], model_ret_for_regression['ind_26'], model_ret_for_regression['ind_27']))
        #
        #
        # zj_cursor.executemany('replace FM_Cleaned_Data (SecuCode, TradingDay, log_ret,  \
        #                           sqrtmarketcap, marketcap, country, beta, momentum, size, \
        #                           earnings_yield, residual_vol, growth, book_to_price, leverage, \
        #                           liquidity, non_linear_size, ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, \
        #                           ind_8, ind_9, ind_10, ind_11, ind_12, ind_13, ind_14, ind_15, ind_16, \
        #                           ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, \
        #                           ind_25, ind_26, ind_27) value (%s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', values)
        #
        #
        # zj_conn.commit()

         #%% Daily Regression


        factor_num = len(factor_list)
        critical_dates = get_critical_date(model_ret_for_regression)
        model_ret_for_regression2 = pd.merge(model_ret_for_regression, critical_dates, on='SecuCode', how='left')

        # Only keep the data where TradingDay is larger than critical dates
        model_ret_for_regression2 = model_ret_for_regression2.loc[model_ret_for_regression2['TradingDay']
                                                                  >= model_ret_for_regression2['CriticalDates'], ]
        logger.info('start to estimate factor returns')
        [wls_beta, wls_constraint, wls_resid, wls_resid_all, wls_r2, wls_adjr2, wls_yhat, wls_yhat_all, wls_tscore] = \
            implied_factor_return(model_ret_for_regression, model_ret_for_regression2,
            'log_ret', factor_list, ind_list, factor_list, factor_num)

        #%% Upload the regression results to database, such as factor returns(wls_beta), residuals,
    #   predicted log return(wls_yhat), R2, and adjusted R2.
        columns = ['TradingDay'] + factor_list
        wls_beta = wls_beta[columns]
        DbHandleUtil.save('FM_WLS_Beta', wls_beta, 'zj_data', has_jsid=False)
        # factor returns
        # values = list(zip(wls_beta['TradingDay'], wls_beta['country'], wls_beta['beta'],
        #               wls_beta['momentum'], wls_beta['size'], wls_beta['earnings_yield'],
        #               wls_beta['residual_vol'], wls_beta['growth'], wls_beta['book_to_price'],
        #               wls_beta['leverage'], wls_beta['liquidity'], wls_beta['non_linear_size'], wls_beta['ind_0'],
        #               wls_beta['ind_1'], wls_beta['ind_2'], wls_beta['ind_3'], wls_beta['ind_4'],
        #               wls_beta['ind_5'], wls_beta['ind_6'], wls_beta['ind_7'], wls_beta['ind_8'],
        #               wls_beta['ind_9'], wls_beta['ind_10'], wls_beta['ind_11'], wls_beta['ind_12'],
        #               wls_beta['ind_13'], wls_beta['ind_14'], wls_beta['ind_15'], wls_beta['ind_16'],
        #               wls_beta['ind_17'], wls_beta['ind_18'], wls_beta['ind_19'], wls_beta['ind_20'],
        #               wls_beta['ind_21'], wls_beta['ind_22'], wls_beta['ind_23'], wls_beta['ind_24'],
        #               wls_beta['ind_25'], wls_beta['ind_26'], wls_beta['ind_27']))
        #
        # zj_cursor.executemany('replace FM_WLS_Beta (TradingDay, country, beta, momentum, size, \
        #                          earnings_yield, residual_vol, growth, book_to_price, leverage, \
        #                          liquidity, non_linear_size, ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, \
        #                          ind_7, ind_8, ind_9, ind_10, ind_11, ind_12, ind_13, ind_14, \
        #                          ind_15, ind_16, ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, \
        #                          ind_23, ind_24, ind_25, ind_26, ind_27) value (%s, %s, %s, %s, \
        #                          %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                          %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                          %s, %s, %s, %s)', values)
        # zj_conn.commit()
        wls_resid_all = wls_resid_all[['TradingDay', 'SecuCode', 'wls_resid', 'wls_resid2']]
        DbHandleUtil.save('FM_WLS_Resid_Var', wls_resid_all, 'zj_data', has_jsid=False)
        # residuals
        # values = list(zip(wls_resid_all['TradingDay'], wls_resid_all['SecuCode'], wls_resid_all['wls_resid'],
        #                   wls_resid_all['wls_resid2']))
        #
        # zj_cursor.executemany('replace FM_WLS_Resid_Var (TradingDay, SecuCode, wls_resid, \
        #                           wls_resid2) value (%s, %s, %s, %s)', values)
        #
        # zj_conn.commit()
        wls_yhat_all = wls_yhat_all[['TradingDay', 'SecuCode', 'wls_yhat']]
        DbHandleUtil.save('FM_WLS_Yhat', wls_yhat_all, 'zj_data', has_jsid=False)
        # predicted log returns
        # values = list(zip(wls_yhat_all['TradingDay'], wls_yhat_all['SecuCode'], wls_yhat_all['wls_yhat']))
        #
        # zj_cursor.executemany('replace FM_WLS_Yhat (TradingDay, SecuCode, wls_yhat) \
        #                           value (%s, %s, %s)', values)
        #
        # zj_conn.commit()
        wls_r2 = wls_r2[['TradingDay', 'r2']]
        DbHandleUtil.save('FM_WLS_R2', wls_r2, 'zj_data', has_jsid=False)
        # R2
        # values = list(zip(wls_r2['TradingDay'], wls_r2['r2']))
        #
        # zj_cursor.executemany('replace FM_WLS_R2 (TradingDay, r2) \
        #                           value (%s, %s)', values)
        #
        # zj_conn.commit()
        wls_adjr2 = wls_adjr2[['TradingDay', 'adjr2']]
        DbHandleUtil.save('FM_WLS_AdjR2', wls_adjr2, 'zj_data', has_jsid=False)
        # Adjusted R2
        # values = list(zip(wls_adjr2['TradingDay'], wls_adjr2['adjr2']))
        #
        # zj_cursor.executemany('replace FM_WLS_AdjR2 (TradingDay, adjr2) \
        #                           value (%s, %s)', values)
        #
        # zj_conn.commit()



    #%% Calculate and upload predicted factor return volatility

        PreBetaVol, PreBetaVol_5d, PreBetaVol_5p, forecasted_factor_return = factor_return_vol_forecast(
            tradingdays, factor_list)
        columns = ['TradingDay'] + factor_list
        PreBetaVol = PreBetaVol[columns]
        DbHandleUtil.save('FM_PreBetaVol', PreBetaVol, 'zj_data', has_jsid=False)

        # values = list(zip(PreBetaVol['TradingDay'], PreBetaVol['country'], PreBetaVol['beta'],
        #                   PreBetaVol['momentum'], PreBetaVol['size'], PreBetaVol['earnings_yield'],
        #                   PreBetaVol['residual_vol'], PreBetaVol['growth'], PreBetaVol['book_to_price'],
        #                   PreBetaVol['leverage'], PreBetaVol['liquidity'], PreBetaVol['non_linear_size'], PreBetaVol['ind_0'],
        #                   PreBetaVol['ind_1'], PreBetaVol['ind_2'], PreBetaVol['ind_3'],
        #                   PreBetaVol['ind_4'], PreBetaVol['ind_5'], PreBetaVol['ind_6'],
        #                   PreBetaVol['ind_7'], PreBetaVol['ind_8'], PreBetaVol['ind_9'],
        #                   PreBetaVol['ind_10'], PreBetaVol['ind_11'], PreBetaVol['ind_12'],
        #                   PreBetaVol['ind_13'], PreBetaVol['ind_14'], PreBetaVol['ind_15'],
        #                   PreBetaVol['ind_16'], PreBetaVol['ind_17'], PreBetaVol['ind_18'],
        #                   PreBetaVol['ind_19'], PreBetaVol['ind_20'], PreBetaVol['ind_21'],
        #                   PreBetaVol['ind_22'], PreBetaVol['ind_23'], PreBetaVol['ind_24'],
        #                   PreBetaVol['ind_25'], PreBetaVol['ind_26'], PreBetaVol['ind_27']))
        #
        # zj_cursor.executemany('replace FM_PreBetaVol (TradingDay, \
        #                           country, beta, momentum, size, earnings_yield, residual_vol,\
        #                           growth, book_to_price, leverage, liquidity, non_linear_size, ind_0, ind_1, \
        #                           ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, \
        #                           ind_10, ind_11, ind_12, ind_13, ind_14, ind_15, ind_16, \
        #                           ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, \
        #                           ind_25, ind_26, ind_27) value (%s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', values)

        PreBetaVol_5d = PreBetaVol_5d[columns]
        DbHandleUtil.save('FM_PreBetaVol_5D', PreBetaVol_5d, 'zj_data', has_jsid=False)
        # values = list(zip(PreBetaVol_5d['TradingDay'], PreBetaVol_5d['country'], PreBetaVol_5d['beta'],
        #                   PreBetaVol_5d['momentum'], PreBetaVol_5d['size'], PreBetaVol_5d['earnings_yield'],
        #                   PreBetaVol_5d['residual_vol'], PreBetaVol_5d['growth'], PreBetaVol_5d['book_to_price'],
        #                   PreBetaVol_5d['leverage'], PreBetaVol_5d['liquidity'], PreBetaVol_5d['non_linear_size'], PreBetaVol_5d['ind_0'],
        #                   PreBetaVol_5d['ind_1'], PreBetaVol_5d['ind_2'], PreBetaVol_5d['ind_3'],
        #                   PreBetaVol_5d['ind_4'], PreBetaVol_5d['ind_5'], PreBetaVol_5d['ind_6'],
        #                   PreBetaVol_5d['ind_7'], PreBetaVol_5d['ind_8'], PreBetaVol_5d['ind_9'],
        #                   PreBetaVol_5d['ind_10'], PreBetaVol_5d['ind_11'], PreBetaVol_5d['ind_12'],
        #                   PreBetaVol_5d['ind_13'], PreBetaVol_5d['ind_14'], PreBetaVol_5d['ind_15'],
        #                   PreBetaVol_5d['ind_16'], PreBetaVol_5d['ind_17'], PreBetaVol_5d['ind_18'],
        #                   PreBetaVol_5d['ind_19'], PreBetaVol_5d['ind_20'], PreBetaVol_5d['ind_21'],
        #                   PreBetaVol_5d['ind_22'], PreBetaVol_5d['ind_23'], PreBetaVol_5d['ind_24'],
        #                   PreBetaVol_5d['ind_25'], PreBetaVol_5d['ind_26'], PreBetaVol_5d['ind_27']))
        #
        # zj_cursor.executemany('replace FM_PreBetaVol_5D (TradingDay, \
        #                           country, beta, momentum, size, earnings_yield, residual_vol,\
        #                           growth, book_to_price, leverage, liquidity, non_linear_size, ind_0, ind_1, \
        #                           ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, \
        #                           ind_10, ind_11, ind_12, ind_13, ind_14, ind_15, ind_16, \
        #                           ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, \
        #                           ind_25, ind_26, ind_27) value (%s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', values)
        columns = ['TradingDay', 'forecast_period'] + factor_list
        PreBetaVol_5p = PreBetaVol_5p[columns]
        DbHandleUtil.save('FM_PreBetaVol_P5', PreBetaVol_5p, 'zj_data', has_jsid=False)
        # values = list(zip(PreBetaVol_5p['TradingDay'], PreBetaVol_5p['forecast_period'], PreBetaVol_5p['country'], PreBetaVol_5p['beta'],
        #                   PreBetaVol_5p['momentum'], PreBetaVol_5p['size'], PreBetaVol_5p['earnings_yield'],
        #                   PreBetaVol_5p['residual_vol'], PreBetaVol_5p['growth'], PreBetaVol_5p['book_to_price'],
        #                   PreBetaVol_5p['leverage'], PreBetaVol_5p['liquidity'], PreBetaVol_5p['non_linear_size'], PreBetaVol_5p['ind_0'],
        #                   PreBetaVol_5p['ind_1'], PreBetaVol_5p['ind_2'], PreBetaVol_5p['ind_3'],
        #                   PreBetaVol_5p['ind_4'], PreBetaVol_5p['ind_5'], PreBetaVol_5p['ind_6'],
        #                   PreBetaVol_5p['ind_7'], PreBetaVol_5p['ind_8'], PreBetaVol_5p['ind_9'],
        #                   PreBetaVol_5p['ind_10'], PreBetaVol_5p['ind_11'], PreBetaVol_5p['ind_12'],
        #                   PreBetaVol_5p['ind_13'], PreBetaVol_5p['ind_14'], PreBetaVol_5p['ind_15'],
        #                   PreBetaVol_5p['ind_16'], PreBetaVol_5p['ind_17'], PreBetaVol_5p['ind_18'],
        #                   PreBetaVol_5p['ind_19'], PreBetaVol_5p['ind_20'], PreBetaVol_5p['ind_21'],
        #                   PreBetaVol_5p['ind_22'], PreBetaVol_5p['ind_23'], PreBetaVol_5p['ind_24'],
        #                   PreBetaVol_5p['ind_25'], PreBetaVol_5p['ind_26'], PreBetaVol_5p['ind_27']))
        #
        # zj_cursor.executemany('replace FM_PreBetaVol_P5 (TradingDay, forecast_period,\
        #                           country, beta, momentum, size, earnings_yield, residual_vol,\
        #                           growth, book_to_price, leverage, liquidity, non_linear_size, ind_0, ind_1, \
        #                           ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, \
        #                           ind_10, ind_11, ind_12, ind_13, ind_14, ind_15, ind_16, \
        #                           ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, \
        #                           ind_25, ind_26, ind_27) value (%s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', values)
        columns = ['TradingDay'] + factor_list
        forecasted_factor_return = forecasted_factor_return[columns]
        DbHandleUtil.save('FM_factor_return_forecast', forecasted_factor_return, 'zj_data', has_jsid=False)
        # values = list(zip(forecasted_factor_return['TradingDay'], forecasted_factor_return['country'], forecasted_factor_return['beta'],
        #                   forecasted_factor_return['momentum'], forecasted_factor_return['size'], forecasted_factor_return['earnings_yield'],
        #                   forecasted_factor_return['residual_vol'], forecasted_factor_return['growth'], forecasted_factor_return['book_to_price'],
        #                   forecasted_factor_return['leverage'], forecasted_factor_return['liquidity'], forecasted_factor_return['non_linear_size'],
        #                   forecasted_factor_return['ind_0'], forecasted_factor_return['ind_1'], forecasted_factor_return['ind_2'],
        #                   forecasted_factor_return['ind_3'], forecasted_factor_return['ind_4'], forecasted_factor_return['ind_5'],
        #                   forecasted_factor_return['ind_6'], forecasted_factor_return['ind_7'], forecasted_factor_return['ind_8'],
        #                   forecasted_factor_return['ind_9'], forecasted_factor_return['ind_10'], forecasted_factor_return['ind_11'],
        #                   forecasted_factor_return['ind_12'], forecasted_factor_return['ind_13'], forecasted_factor_return['ind_14'],
        #                   forecasted_factor_return['ind_15'], forecasted_factor_return['ind_16'], forecasted_factor_return['ind_17'],
        #                   forecasted_factor_return['ind_18'], forecasted_factor_return['ind_19'], forecasted_factor_return['ind_20'],
        #                   forecasted_factor_return['ind_21'], forecasted_factor_return['ind_22'], forecasted_factor_return['ind_23'],
        #                   forecasted_factor_return['ind_24'], forecasted_factor_return['ind_25'], forecasted_factor_return['ind_26'],
        #                   forecasted_factor_return['ind_27']))
        #
        # zj_cursor.executemany('replace FM_factor_return_forecast (TradingDay, \
        #                           country, beta, momentum, size, earnings_yield, residual_vol,\
        #                           growth, book_to_price, leverage, liquidity, non_linear_size, ind_0, ind_1, \
        #                           ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, \
        #                           ind_10, ind_11, ind_12, ind_13, ind_14, ind_15, ind_16, \
        #                           ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, \
        #                           ind_25, ind_26, ind_27) value (%s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', values)
        # zj_conn.commit()
#        residual_variance_forecast(zj_conn, zj_cursor, tradingdays)
#        error = calculate_risk_forecast(zj_conn, jy_conn, zj_cursor, tradingdays)
        factor_correlation = factor_return_correlation(tradingdays)
        columns = ['TradingDay', 'factor'] + factor_list
        factor_correlation = factor_correlation[columns]
        DbHandleUtil.save('FM_factor_return_correlation', factor_correlation, 'zj_data', has_jsid=False)
        # values = list(zip(factor_correlation['TradingDay'], factor_correlation['factor'], factor_correlation['country'], factor_correlation['beta'],
        #                   factor_correlation['momentum'], factor_correlation['size'], factor_correlation['earnings_yield'],
        #                   factor_correlation['residual_vol'], factor_correlation['growth'], factor_correlation['book_to_price'],
        #                   factor_correlation['leverage'], factor_correlation['liquidity'], factor_correlation['non_linear_size'],
        #                   factor_correlation['ind_0'], factor_correlation['ind_1'], factor_correlation['ind_2'],
        #                   factor_correlation['ind_3'], factor_correlation['ind_4'], factor_correlation['ind_5'],
        #                   factor_correlation['ind_6'], factor_correlation['ind_7'], factor_correlation['ind_8'],
        #                   factor_correlation['ind_9'], factor_correlation['ind_10'], factor_correlation['ind_11'],
        #                   factor_correlation['ind_12'], factor_correlation['ind_13'], factor_correlation['ind_14'],
        #                   factor_correlation['ind_15'], factor_correlation['ind_16'], factor_correlation['ind_17'],
        #                   factor_correlation['ind_18'], factor_correlation['ind_19'], factor_correlation['ind_20'],
        #                   factor_correlation['ind_21'], factor_correlation['ind_22'], factor_correlation['ind_23'],
        #                   factor_correlation['ind_24'], factor_correlation['ind_25'], factor_correlation['ind_26'],
        #                   factor_correlation['ind_27']))
        #
        # zj_cursor.executemany('replace FM_factor_return_correlation (TradingDay, factor, \
        #                           country, beta, momentum, size, earnings_yield, residual_vol,\
        #                           growth, book_to_price, leverage, liquidity, non_linear_size, ind_0, ind_1, \
        #                           ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, \
        #                           ind_10, ind_11, ind_12, ind_13, ind_14, ind_15, ind_16, \
        #                           ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, \
        #                           ind_25, ind_26, ind_27) value (%s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', values)
        #
        # zj_conn.commit()

        factor_correlation_ewma = factor_return_correlation_ewma(tradingdays)
        columns = ['TradingDay', 'factor'] + factor_list
        factor_correlation_ewma = factor_correlation_ewma[columns]
        DbHandleUtil.save('FM_factor_return_correlation_ewma', factor_correlation_ewma, 'zj_data', has_jsid=False)
        # values = list(zip(factor_correlation_ewma['TradingDay'], factor_correlation_ewma['factor'], factor_correlation_ewma['country'], factor_correlation_ewma['beta'],
        #                   factor_correlation_ewma['momentum'], factor_correlation_ewma['size'], factor_correlation_ewma['earnings_yield'],
        #                   factor_correlation_ewma['residual_vol'], factor_correlation_ewma['growth'], factor_correlation_ewma['book_to_price'],
        #                   factor_correlation_ewma['leverage'], factor_correlation_ewma['liquidity'], factor_correlation_ewma['non_linear_size'],
        #                   factor_correlation_ewma['ind_0'], factor_correlation_ewma['ind_1'], factor_correlation_ewma['ind_2'],
        #                   factor_correlation_ewma['ind_3'], factor_correlation_ewma['ind_4'], factor_correlation_ewma['ind_5'],
        #                   factor_correlation_ewma['ind_6'], factor_correlation_ewma['ind_7'], factor_correlation_ewma['ind_8'],
        #                   factor_correlation_ewma['ind_9'], factor_correlation_ewma['ind_10'], factor_correlation_ewma['ind_11'],
        #                   factor_correlation_ewma['ind_12'], factor_correlation_ewma['ind_13'], factor_correlation_ewma['ind_14'],
        #                   factor_correlation_ewma['ind_15'], factor_correlation_ewma['ind_16'], factor_correlation_ewma['ind_17'],
        #                   factor_correlation_ewma['ind_18'], factor_correlation_ewma['ind_19'], factor_correlation_ewma['ind_20'],
        #                   factor_correlation_ewma['ind_21'], factor_correlation_ewma['ind_22'], factor_correlation_ewma['ind_23'],
        #                   factor_correlation_ewma['ind_24'], factor_correlation_ewma['ind_25'], factor_correlation_ewma['ind_26'],
        #                   factor_correlation_ewma['ind_27']))
        #
        # zj_cursor.executemany('replace FM_factor_return_correlation_ewma (TradingDay, factor, \
        #                           country, beta, momentum, size, earnings_yield, residual_vol,\
        #                           growth, book_to_price, leverage, liquidity, non_linear_size, ind_0, ind_1, \
        #                           ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, \
        #                           ind_10, ind_11, ind_12, ind_13, ind_14, ind_15, ind_16, \
        #                           ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, \
        #                           ind_25, ind_26, ind_27) value (%s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
        #                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', values)
        #
        # zj_conn.commit()


    # zj_cursor.close()
    # jy_cursor.close()

