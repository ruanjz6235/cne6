import pandas as pd
import numpy as np
from app.utils.DbUtil import DbUtil
from app.utils.HiveUtil import HiveUtil
import jaydebeapi as jdb
from app.model.v5.data.base import get_tradingdays


def dws_var_info_indx_quot_dd(code):
    conn = jdb.connect(jclassname="org.apache.hive.jdbc.HiveDriver",
                       url="jdbc:hive2://10.55.136.88:21050/riskdws",
                       driver_args=['nltosql', 'XUG9RMAnUB%MNkRXm7Yl'],
                       jars="/home/WUYING_wangrunze_1985371252/文档/QIMPro/My Received Files/364/hive-jdbc-2.1.1-cdh6.3.2-standalone.jar")
    query = f"""
    SELECT
        idx.busi_date end_date,
    
        -- 2. Volatility & Momentum Factors (波动率与动量因子)
        idx.tdy_clqn_prc close,       -- 收盘价 (用于 Beta/RSTR 计算)
        idx.tdy_clqn_prc/idx.yest_clqn_prc  index_return

    FROM
        -- 4. 指数行情表 (用于获取基准收益率，例如 000300.SH)
        riskdws.dws_var_info_indx_quot_dd idx
    WHERE idx.scr_code = '{code}'
    ORDER BY idx.busi_date asc
    """
    a = pd.read_sql(query, conn)
    return a


def dm_inrs_ins_fin_indx_dd(start, end):
    conn = jdb.connect(jclassname="org.apache.hive.jdbc.HiveDriver",
                       url="jdbc:hive2://10.55.136.88:21050/riskdws",
                       driver_args=['nltosql', 'XUG9RMAnUB%MNkRXm7Yl'],
                       jars="/home/WUYING_wangrunze_1985371252/文档/QIMPro/My Received Files/364/hive-jdbc-2.1.1-cdh6.3.2-standalone.jar")
    query = f"""
    select fin.busi_date end_date, fin.rela_date pub_date, fin.rept_data_date report_date, fin.ins_num,
        -- 4. Value Factor (价值因子)
        fin.net_prof_ttm,                          -- 净利润 TTM (用于 EPTTM)
        fin.ebit_ttm,                              -- 息税前利润 TTM (用于 EBIT_EV)
    
        -- 5. Quality Factor (质量因子)
        fin.liab_tot AS total_liability,           -- 总负债 (用于 DTOA)
        fin.tot_ast AS total_asset,                -- 总资产 (用于 DTOA/AT/GP/ROA 等)
        fin.busi_tot_incm_ttm AS total_income_ttm, -- 营业总收入 TTM (用于 AT/GP/VOS)
        fin.busi_tot_cost_ttm AS total_cost_ttm,   -- 营业总成本 TTM (用于 GP/GPM)
        fin.net_cash_oper_ttm AS cash_flow_ttm,    -- 经营性现金流 TTM (用于 VOC)
        fin.net_cash_ivsm_ttm AS capex,            -- 投资性现金流 (代理资本支出用于 CEG)
        -- 注意：accr_bs (应计项目) 在字典中未明确，通常可使用 (净利润 - 经营现金流) 代理

        (fin.net_prof_ttm - fin.net_cash_oper_ttm) AS accr_bs,
    
        -- 6. Growth Factor (成长因子)
        fin.ps_busi_incm AS income_ps,             -- 每股营业收入 (用于 SGRO)
        fin.eps                                   -- 每股收益 (用于 EGRO)
    
    from riskdm.dm_inrs_ins_fin_indx_dd fin
    inner join riskdm.dm_inrs_stk_perf_dd perf on 
    where busi_date > '{start}'
    and busi_date <= '{end}'
    order by busi_date asc
    """
    a = pd.read_sql(query, conn)
    return a


def dm_inrs_stk_perf_dd(start, end):
    conn = jdb.connect(jclassname="org.apache.hive.jdbc.HiveDriver",
                       url="jdbc:hive2://10.55.136.88:21050/riskdws",
                       driver_args=['nltosql', 'XUG9RMAnUB%MNkRXm7Yl'],
                       jars="/home/WUYING_wangrunze_1985371252/文档/QIMPro/My Received Files/364/hive-jdbc-2.1.1-cdh6.3.2-standalone.jar")
    query = f"""
    SELECT
        perf.busi_date end_date, perf.scr_code secu_code, perf.ins_num,
    
        -- 1. Size Factor (尺寸因子)
        perf.tot_mval AS market_value,             -- 总市值 (用于 LNCAP 计算)
    
        -- 2. Volatility & Momentum Factors (波动率与动量因子)
        perf.plmt AS pct_change,                   -- 个股日收益率
        perf.tdy_clqn_prc AS close,                -- 今日收盘价 (用于 CMRA/PRED_VOEP)
    
        -- 3. Dividend Yield Factor (红利因子)
        perf.divd_rate_ttm,                        -- 股息率 TTM
    
        -- 4. Value Factor (价值因子)
        CASE WHEN perf.pb > 0 THEN 1.0 / perf.pb ELSE NULL END AS bp,
        perf.corp_val_crcp,                         -- 企业价值 (用于 EBIT_EV 分母)
    
        -- 5. Quality Factor (质量因子)
        perf.cir_capt AS circulate_shares,         -- 流通股本 (用于 IG 增长计算)
    
        -- 7. Liquidity Factor (流动性因子)
        perf.mth_tnrt,                             -- 月换手率 (STOM)
        perf.quar_tnrt,                            -- 季换手率 (STOQ)
        perf.ann_tnrt AS ann_tnrt                  -- 年换手率 (STOA)
    
    FROM
        -- 1. 股票日频业绩表现表 (主驱动表)
        riskdm.dm_inrs_stk_perf_dd perf
    WHERE perf.busi_date > '{start}'
      AND perf.busi_date <= '{end}'
      AND perf.tdy_clqn_prc IS NOT NULL
      and perf.astk_boar_type_name in ('主板', '创业板', '科创板')
    ORDER BY perf.busi_date asc, perf.scr_code
    """
    a = pd.read_sql(query, conn)
    return a


def dm_inrs_stk_prof_cmph_perd_dd(start, end):
    conn = jdb.connect(jclassname="org.apache.hive.jdbc.HiveDriver",
                       url="jdbc:hive2://10.55.136.88:21050/riskdws",
                       driver_args=['nltosql', 'XUG9RMAnUB%MNkRXm7Yl'],
                       jars="/home/WUYING_wangrunze_1985371252/文档/QIMPro/My Received Files/364/hive-jdbc-2.1.1-cdh6.3.2-standalone.jar")
    query = f"""
    SELECT
        prof.busi_date end_date, prof.scr_code secu_code,
    
        -- 3. Dividend Yield Factor (红利因子)
        prof.pred_divd_rate_avg,                   -- 预测股息率
    
        -- 4. Value Factor (价值因子)
        prof.pred_pera_avg,                        -- 分析师预测 E/P (预测市盈率倒数)
    
        -- 5. Quality Factor (质量因子)
        prof.pred_eps_stde AS pred_eps_std,        -- 预测 EPS 标准差 (用于 PRED_VOEP)
    
        -- 6. Growth Factor (成长因子)
        prof.pred_net_prof_gtrt                    -- 预测净利润增长率 (用于 EGRLF)
    
    FROM
        -- 2. 盈利预测综合表 (不定期预测)
        riskdm.dm_inrs_stk_prof_cmph_perd_dd prof
    WHERE prof.busi_date > '{start}'
      AND prof.busi_date <= '{end}'
      and prof.astk_boar_type_name in ('主板', '创业板', '科创板')
    ORDER BY prof.busi_date asc, prof.scr_code
    """
    a = pd.read_sql(query, conn)
    return a


if __name__ == '__main__':
    from glob import glob
    lst = glob(r'D:\claude\Barra\cne6\data\barra_test\dm_inrs_ins_fin_indx_dd_*.feather')
    a_s = []
    for file in lst:
        print(file[26: 34])
        if file[26:34] < '20120430':
            a_s.append(pd.read_feather(file))
    a = pd.read_feather(r'dm_inrs_ins_fin_indx_dd_20091231_20100131.feather')
    start_date, end_date = '2009-12-31', '2026-04-17'
    tradingdays = get_tradingdays(start_date=start_date, end_date=end_date, if_hk=0)
    tradingdays['end_date'] = tradingdays['end_date'].astype(str).str[:10].str.replace('-', '').astype(int)

    index_price = dws_var_info_indx_quot_dd('000300')
    index_price = tradingdays.merge(index_price, on=['end_date'], how='inner').reset_index(drop=True)
    index_price.to_feather(f'dws_var_info_indx_quot_dd.feather')

    for start in pd.date_range(start_date, end_date, freq='Y'):
        end = start + pd.DateOffset(months=1)
        fin_data = dm_inrs_ins_fin_indx_dd(start, end)
        fin_data = fin_data.merge(tradingdays, on=['end_date'], how='inner')
        fin_data.to_feather(f'dm_inrs_ins_fin_indx_dd_{start}_{end}.feather')

    for start in pd.date_range(start_date, end_date, freq='Y'):
        end = start + pd.DateOffset(months=1)
        perf_data = dm_inrs_stk_perf_dd(start, end)
        perf_data = perf_data.merge(tradingdays, on=['end_date'], how='inner')
        perf_data.to_feather(f'dm_inrs_stk_perf_dd_{start}_{end}.feather')

    for start in pd.date_range(start_date, end_date, freq='Y'):
        end = start + pd.DateOffset(months=1)
        perf_data = dm_inrs_stk_prof_cmph_perd_dd(start, end)
        perf_data = perf_data.merge(tradingdays, on=['end_date'], how='inner')
        perf_data.to_feather(f'dm_inrs_stk_prof_cmph_perd_dd_{start}_{end}.feather')




