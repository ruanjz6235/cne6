"""
Barra数据定时更新任务
每日凌晨执行，更新以下数据文件:
1. dws_var_info_indx_quot_dd.feather - 指数行情全量更新
2. dm_inrs_ins_fin_indx_dd_{月始}_{月终}.feather - 财务指标增量更新
3. dm_inrs_stk_prof_cmph_perd_dd_{月始}_{月终}.feather - 盈利预测增量更新
"""

import pandas as pd
import os
from glob import glob
from datetime import datetime
from app.model.v5.data.base import get_tradingdays
# import jaydebeapi as jdb
from app.utils.HiveUtil import HiveUtil
from app.common.log import logger

# 数据存储目录
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# # Hive数据库连接配置
# HIVE_CONFIG = {
#     'jclassname': "org.apache.hive.jdbc.HiveDriver",
#     'url': "jdbc:hive2://10.55.136.88:21050/riskdws",
#     'driver_args': ['nltosql', 'XUG9RMAnUB%MNkRXm7Yl'],
#     "jars": "/home/WUYING_ruanjiazheng_1985371/文档/QIMPro/My Received Files/362/hive-jdbc-2.1.1-cdh6.3.2-standalone.jar"
# }


# def get_hive_connection():
#     """获取Hive数据库连接"""
#     return jdb.connect(**HIVE_CONFIG)


def dws_var_info_indx_quot_dd(code):
    """查询指数行情数据"""
    # conn = get_hive_connection()
    conn = HiveUtil.get_conn('riskdws')
    query = f"""
    SELECT
        idx.pt_date as end_date,

        -- 2. Volatility & Momentum Factors (波动率与动量因子)
        idx.tdy_clqn_prc as close,       -- 收盘价 (用于 Beta/RSTR 计算)
        idx.tdy_clqn_prc/idx.yest_clqn_prc as index_return

    FROM
        -- 4. 指数行情表 (用于获取基准收益率，例如 000300.SH)
        riskdws.dws_var_info_indx_quot_dd idx
    WHERE idx.scr_code = '{code}'
    and mkt_code = 'OTC'
    ORDER BY idx.busi_date asc
    """
    data = pd.read_sql(query, conn)
    return data


def dm_inrs_ins_fin_indx_dd(start, end):
    """查询财务指标数据"""
    # conn = get_hive_connection()
    conn = HiveUtil.get_conn('riskdws')
    query = f"""
    select fin.pt_date as end_date, fin.rela_date as rela_date, fin.rept_data_date as rept_data_date,
           fin.ins_num as ins_num, perf.scr_code as secu_code,
        -- 4. Value Factor (价值因子)
        fin.net_prof_ttm as net_prof_ttm,                      -- 净利润 TTM (用于 EPTTM)
        fin.ebit_ttm as ebit_ttm,                              -- 息税前利润 TTM (用于 EBIT_EV)

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
        fin.eps as eps,                            -- 每股收益 (用于 EGRO)

        -- 1. Size Factor (尺寸因子)
        perf.tot_mval AS market_value,             -- 总市值 (用于 LNCAP 计算)

        -- 2. Volatility & Momentum Factors (波动率与动量因子)
        perf.plmt AS pct_change,                   -- 个股日收益率
        perf.tdy_clqn_prc AS close,                -- 今日收盘价 (用于 CMRA/PRED_VOEP)

        -- 3. Dividend Yield Factor (红利因子)
        perf.divd_rate_ttm as divd_rate_ttm,                        -- 股息率 TTM

        -- 4. Value Factor (价值因子)
        CASE WHEN perf.pb > 0 THEN 1.0 / perf.pb ELSE NULL END AS bp,
        perf.corp_val_crcp as corp_val_crcp,                         -- 企业价值 (用于 EBIT_EV 分母)

        -- 5. Quality Factor (质量因子)
        perf.cir_capt AS circulate_shares,         -- 流通股本 (用于 IG 增长计算)

        -- 7. Liquidity Factor (流动性因子)
        perf.mth_tnrt as mth_tnrt,                             -- 月换手率 (STOM)
        perf.quar_tnrt as quar_tnrt,                            -- 季换手率 (STOQ)
        perf.ann_tnrt AS ann_tnrt                  -- 年换手率 (STOA)

    from riskdm.dm_inrs_ins_fin_indx_dd fin
    inner join riskdm.dm_inrs_stk_perf_dd perf
    on fin.pt_date = perf.pt_date and fin.ins_num = perf.ins_num
    where fin.pt_date > '{start}' and fin.pt_date <= '{end}'
    AND perf.tdy_clqn_prc IS NOT NULL
    and perf.astk_boar_type_name in ('主板', '创业板', '科创板')
    order by perf.pt_date asc, perf.scr_code
    """
    # logger.info(query)
    data = pd.read_sql(query, conn)
    return data


def dm_inrs_stk_prof_cmph_perd_dd(start, end):
    """拉取指定交易日的盈利预测数据（单日）"""
    # conn = get_hive_connection()
    conn = HiveUtil.get_conn('riskdws')
    query = f"""
    SELECT
        prof.pt_date as end_date, prof.scr_code secu_code,

        -- 3. Dividend Yield Factor (红利因子)
        prof.pred_divd_rate_avg as pred_divd_rate_avg,                   -- 预测股息率

        -- 4. Value Factor (价值因子)
        prof.pred_pera_avg as pred_pera_avg,                        -- 分析师预测 E/P (预测市盈率倒数)

        -- 5. Quality Factor (质量因子)
        prof.pred_eps_stde AS pred_eps_std,        -- 预测 EPS 标准差 (用于 PRED_VOEP)

        -- 6. Growth Factor (成长因子)
        prof.pred_net_prof_gtrt as pred_net_prof_gtrt                    -- 预测净利润增长率 (用于 EGRLF)

    FROM
        -- 2. 盈利预测综合表 (不定期预测)
        riskdm.dm_inrs_stk_prof_cmph_perd_dd prof
    WHERE prof.pt_date > '{start}'
      AND prof.pt_date <= '{end}'
      and prof.pred_year = {str(pd.to_datetime(end)-pd.DateOffset(days=1)+pd.DateOffset(years=1)+pd.offsets.YearEnd())[:4]}
      and prof.astk_boar_type_name in ('主板', '创业板', '科创板')
    ORDER BY prof.pt_date asc, prof.scr_code
    """
    # logger.info(query)
    data = pd.read_sql(query, conn)
    return data


def update_index_price():
    """更新指数行情数据 - 全量更新"""
    logger.info(f"[{datetime.now()}] 开始更新指数行情数据...")

    today = pd.to_datetime('today')
    start_date = '2009-12-31'
    end_date = today.strftime('%Y-%m-%d')

    tradingdays = get_tradingdays(start_date=start_date, end_date=end_date, if_hk=0)
    tradingdays['end_date'] = tradingdays['end_date'].astype(str).str[:10].str.replace('-', '')

    index_price = dws_var_info_indx_quot_dd('000300')
    index_price = tradingdays.merge(index_price, on=['end_date'], how='inner').reset_index(drop=True)

    file_path = os.path.join(DATA_DIR, 'dws_var_info_indx_quot_dd.feather')
    index_price.to_feather(file_path)

    logger.info(f"[{datetime.now()}] 指数行情数据更新完成，共 {len(index_price)} 条记录")


def update_monthly_data(table_name, query_func, today):
    """
    更新按月存储的数据文件

    Args:
        table_name: 表名前缀
        query_func: 数据查询函数
        today: 当前日期
    """
    file_path = sorted(glob(DATA_DIR + '/' + table_name + '*'))[-1]
    today_str = today.strftime('%Y%m%d')
    existing_data = pd.read_feather(file_path)
    max_date = existing_data['end_date'].max()

    tradingdays = get_tradingdays(
        start_date=pd.to_datetime(max_date),
        end_date=today.normalize(),
        if_hk=0
    )
    tradingdays['end_date'] = tradingdays['end_date'].astype(str).str[:10].str.replace('-', '')

    new_data = query_func(max_date, today_str)
    new_data = tradingdays.merge(new_data, on=['end_date'], how='inner')

    if today.month == pd.to_datetime(max_date).month:
        merged_data = pd.concat([existing_data, new_data], ignore_index=True)
        # merged_data = merged_data.drop_duplicates(
        #     subset=['end_date', 'secu_code'],
        #     keep='last')
        merged_data = merged_data.sort_values(['end_date', 'secu_code']).reset_index(drop=True)
        merged_data.to_feather(file_path)
        logger.info(f"[{datetime.now()}] {table_name} 增量更新完成，新增 {len(new_data)} 条，总计 {len(merged_data)} 条")
    else:
        month_start = today - pd.offsets.MonthEnd()
        month_end = today - pd.DateOffset(days=1) + pd.offsets.MonthEnd()
        new_data1 = new_data[new_data['end_date'] <= month_start.strftime('%Y%m%d')]
        new_data2 = new_data[new_data['end_date'] > month_start.strftime('%Y%m%d')].reset_index(drop=True)
        merged_data1 = pd.concat([existing_data, new_data1]).reset_index(drop=True)
        merged_data1.to_feather(file_path)
        new_data2.to_feather(DATA_DIR + '/' + table_name + f"""_{month_start.strftime('%Y%m%d')}_{month_end.strftime('%Y%m%d')}.feather""")
        logger.info(f"[{datetime.now()}] {table_name} 新建月文件，共 {len(new_data2)} 条记录")


def update_financial_data():
    """更新财务指标数据"""
    logger.info(f"[{datetime.now()}] 开始更新财务指标数据...")
    today = pd.to_datetime('today')
    update_monthly_data('dm_inrs_ins_fin_indx_dd', dm_inrs_ins_fin_indx_dd, today)


def update_profit_forecast():
    """更新盈利预测数据"""
    logger.info(f"[{datetime.now()}] 开始更新盈利预测数据...")
    today = pd.to_datetime('today')
    update_monthly_data('dm_inrs_stk_prof_cmph_perd_dd', dm_inrs_stk_prof_cmph_perd_dd, today)


def update_barra_raw_data():
    """执行每日数据更新任务"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"[{datetime.now()}] 开始执行每日数据更新任务")
    logger.info(f"{'=' * 60}\n")
    if glob(DATA_DIR + '/dm_inrs_ins_fin_indx_dd*') == 0:
        his_barra_raw_data()
        return 'first_success'

    try:
        update_index_price()
        update_financial_data()
        update_profit_forecast()

        logger.info(f"\n[{datetime.now()}] 所有数据更新完成")
    except Exception as e:
        logger.info(f"[{datetime.now()}] 数据更新出错: {str(e)}")
        raise
    return 'update_success'


def his_barra_raw_data():
    start_date, end_date = '2009-12-31', pd.to_datetime('today').strftime('%Y-%m-%d')
    tradingdays = get_tradingdays(start_date=start_date, end_date=end_date, if_hk=0)
    tradingdays['end_date'] = tradingdays['end_date'].astype(str).str[:10].str.replace('-', '')

    # index_price = dws_var_info_indx_quot_dd(code='000300')
    # index_price = tradingdays.merge(index_price, on=['end_date'], how='inner').reset_index(drop=True)
    # index_price.to_feather(f'dws_var_info_indx_quot_dd.feather')

    for start in pd.date_range(start_date, end_date, freq='M'):
        if start < pd.to_datetime('2011-01-01'):
            continue
        end = start + pd.offsets.MonthEnd()
        logger.info(start, end)
        start, end = str(start)[:10].replace('-', ''), str(end)[:10].replace('-', '')
        fin_data = dm_inrs_ins_fin_indx_dd(start, end)
        fin_data = fin_data.merge(tradingdays, on=['end_date'], how='inner')
        fin_data.to_feather(f'dm_inrs_ins_fin_indx_dd_{start}_{end}.feather')

    # for start in pd.date_range(start_date, end_date, freq='M'):
    #     logger.info(start)
    #     end = start + pd.DateOffset(months=1)
    #     start, end = str(start)[:10].replace('-', ''), str(end)[:10].replace('-', '')
    #     perf_data = dm_inrs_stk_perf_dd(start, end)
    #     perf_data = perf_data.merge(tradingdays, on=['end_date'], how='inner')
    #     perf_data.to_feather(f'dm_inrs_stk_perf_dd_{start}_{end}.feather')

    for start in pd.date_range(start_date, end_date, freq='M'):
        end = start + pd.offsets.MonthEnd()
        logger.info(start, end)
        start, end = str(start)[:10].replace('-', ''), str(end)[:10].replace('-', '')
        perf_data = dm_inrs_stk_prof_cmph_perd_dd(start, end)
        perf_data = perf_data.merge(tradingdays, on=['end_date'], how='inner')
        perf_data.to_feather(f'dm_inrs_stk_prof_cmph_perd_dd_{start}_{end}.feather')


if __name__ == '__main__':
    # his_barra_raw_data()
    update_barra_raw_data()
