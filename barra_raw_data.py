"""
Barra CNE6 模型数据取数模块
===========================

提供 Barra 基础数据和行业数据的获取功能。
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime
import pymysql


@dataclass
class RawDataConfig:
    """基础数据配置参数"""
    start_date: str  # 开始日期 YYYYMMDD
    end_date: str  # 结束日期 YYYYMMDD
    market: str = 'A'  # 市场类型：A-A 股


class BarraRawData:
    """
    Barra 基础数据取数类

    提供从数据库获取 Barra 计算所需的基础数据功能。
    包括：个股行情数据、财务数据、盈利预测数据、行业分类数据。
    """

    def __init__(self, config: RawDataConfig):
        """
        初始化取数类

        Parameters
        ----------
        config : RawDataConfig
            数据配置参数
        """
        self.config = config
        self.raw_data: Optional[pd.DataFrame] = None
        self.industry_data: Optional[pd.DataFrame] = None

    # ==================== 数据加载方法 ====================

    def load_raw_data_from_sql(self, connection, query: str) -> pd.DataFrame:
        """
        从 SQL 查询加载基础数据

        Parameters
        ----------
        query : str
            SQL 查询语句
        connection : 数据库连接对象

        Returns
        -------
        pd.DataFrame
            基础数据 DataFrame
        """
        self.raw_data = pd.read_sql(query, connection)
        return self._process_raw_data(self.raw_data)

    def load_raw_data_from_pickle(self, file_path: str) -> pd.DataFrame:
        """
        从 pickle 文件加载基础数据

        Parameters
        ----------
        file_path : str
            pickle 文件路径

        Returns
        -------
        pd.DataFrame
            基础数据 DataFrame
        """
        self.raw_data = pd.read_pickle(file_path)
        return self._process_raw_data(self.raw_data)

    def load_raw_data_from_feather(self, file_path: str) -> pd.DataFrame:
        """
        从 feather 文件加载基础数据

        Parameters
        ----------
        file_path : str
            feather 文件路径

        Returns
        -------
        pd.DataFrame
            基础数据 DataFrame
        """
        self.raw_data = pd.read_feather(file_path)
        return self._process_raw_data(self.raw_data)

    def load_raw_data_from_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        从 CSV 文件加载基础数据

        Parameters
        ----------
        file_path : str
            CSV 文件路径
        **kwargs : 传递给 pd.read_csv 的其他参数

        Returns
        -------
        pd.DataFrame
            基础数据 DataFrame
        """
        self.raw_data = pd.read_csv(file_path, **kwargs)
        return self._process_raw_data(self.raw_data)

    def load_industry_data_from_sql(self, query: str, connection) -> pd.DataFrame:
        """
        从 SQL 查询加载行业数据

        Parameters
        ----------
        query : str
            SQL 查询语句（参考 lc_exgindustry_a_share_latest.sql）
        connection : 数据库连接对象

        Returns
        -------
        pd.DataFrame
            行业分类数据 DataFrame
        """
        self.industry_data = pd.read_sql(query, connection)
        return self._process_industry_data(self.industry_data)

    def load_industry_data_from_pickle(self, file_path: str) -> pd.DataFrame:
        """
        从 pickle 文件加载行业数据

        Parameters
        ----------
        file_path : str
            pickle 文件路径

        Returns
        -------
        pd.DataFrame
            行业分类数据 DataFrame
        """
        self.industry_data = pd.read_pickle(file_path)
        return self._process_industry_data(self.industry_data)

    def load_industry_data_from_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        从 CSV 文件加载行业数据

        Parameters
        ----------
        file_path : str
            CSV 文件路径
        **kwargs : 传递给 pd.read_csv 的其他参数

        Returns
        -------
        pd.DataFrame
            行业分类数据 DataFrame
        """
        self.industry_data = pd.read_csv(file_path, **kwargs)
        return self._process_industry_data(self.industry_data)

    # ==================== 数据预处理方法 ====================

    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理基础数据：标准化列名、数据类型转换、日期筛选

        Parameters
        ----------
        df : pd.DataFrame
            原始数据

        Returns
        -------
        pd.DataFrame
            处理后的数据
        """
        processed_df = df.copy()

        # 列名映射（数据库字段 -> 统一字段名）
        column_mapping = {
            # 基础信息
            'secu_code': 'secu_code',
            'stock_code': 'secu_code',
            'scr_code': 'secu_code',
            'code': 'secu_code',

            'secu_abbr': 'secu_name',
            'stock_name': 'secu_name',
            'scr_name': 'secu_name',

            'company_code': 'company_code',

            'trade_date': 'end_date',
            'busi_date': 'end_date',
            'end_date': 'end_date',
            'rept_data_date': 'rept_data_date',

            # 行情数据
            'close': 'close',
            'tdy_clqn_prc': 'close',
            'yest_clqn_prc': 'prev_close',

            'volume': 'volume',
            'mtch_vol': 'volume',

            'amount': 'amount',
            'mtch_amt': 'amount',

            'pct_change': 'pct_change',

            # 市值数据
            'market_value': 'market_value',
            'tot_mval': 'market_value',

            'circle_capital': 'circulate_shares',
            'cir_capt': 'circulate_shares',

            # 估值数据
            'pb': 'pb',
            'pe': 'pe',

            # 股息率
            'dividend_yield_ttm': 'divide_rate_ttm',
            'divd_rate_ttm': 'divide_rate_ttm',
            'pred_divd_rate': 'pred_divide_rate',

            # 换手率
            'month_turnover': 'month_turnover',
            'mth_tnrt': 'month_turnover',
            'quarter_turnover': 'quarter_turnover',
            'quar_tnrt': 'quarter_turnover',
            'annual_turnover': 'annual_turnover',
            'ann_tnrt': 'annual_turnover',

            # 盈利预测
            'pred_net_profit_yoy': 'pred_net_profit_yoy',
            'pred_net_prof_gtrt': 'pred_net_profit_yoy',
            'pred_eps_std': 'pred_eps_std',
            'pred_pe_avg': 'pred_pe_avg',
            'analyst_forecast_profit_margin': 'pred_net_profit_yoy',

            # 财务数据
            'total_asset': 'total_asset',
            'tot_ast': 'total_asset',

            'total_liability': 'total_liability',
            'liab_tot': 'total_liability',

            'total_equity': 'total_equity',

            'total_income': 'total_income',
            'total_income_ttm': 'total_income_ttm',
            'busi_tot_incm': 'total_income',
            'busi_tot_incm_ttm': 'total_income_ttm',

            'total_cost': 'total_cost',
            'total_cost_ttm': 'total_cost_ttm',

            'net_profit': 'net_profit',
            'net_profit_ttm': 'net_profit_ttm',
            'fi.net_prof': 'net_profit',

            'income_ps': 'income_ps',
            'income_ttm_ps': 'income_ttm_ps',

            'eps': 'eps',
            'eps_ttm': 'eps_ttm',

            'cash': 'cash',
            'net_cash_operate': 'net_cash_operate',
            'net_cash_operate_ttm': 'net_cash_operate_ttm',
            'net_cash_invest': 'net_cash_invest',
            'net_cash_invest_ttm': 'net_cash_invest_ttm',

            'ebit': 'ebit',
            'ebit_ttm': 'ebit_ttm',

            'long_liability_divide_total_equity': 'long_liability_divide_total_equity',
            'not_liquid_liability_tot': 'not_liquid_liability_tot',
            'debt_with_interest': 'debt_with_interest',

            'current_deperation': 'current_deperation',
            'corporation_value': 'corporation_value',
            'corporation_value_with_cash': 'corporation_value_with_cash',

            # Value/Quality 因子新增字段
            'corp_val_crcp': 'corp_val_crcp',             # 企业价值 (EV)
            'enterprise_value': 'corp_val_crcp',          # 企业价值别名
            'pred_net_prof_ttm': 'pred_net_profit_ttm',   # 预期净利润TTM
            'pred_eps_std': 'pred_eps_std',               # 预期EPS标准差
            'accr_bs': 'accr_bs',                         # 资产负债表应计项目
            'capital_expenditure': 'capex',               # 资本支出
            'capex': 'capex',                             # 资本支出别名
            'cash_flow': 'cash_flow',                     # 现金流 (年报)

            # 指数数据
            'index_abbr': 'index_abbr',
            'index_close': 'index_close',
            'hs300_close': 'index_close',

            'pct_change_tresure_1y': 'rf',  # 1 年期国债收益率作为无风险利率
        }

        # 重命名列
        rename_dict = {}
        for col in processed_df.columns:
            col_lower = col.lower()
            if col_lower in column_mapping:
                rename_dict[col] = column_mapping[col_lower]

        if rename_dict:
            processed_df = processed_df.rename(columns=rename_dict)

        # 确保必要的列存在
        required_columns = ['secu_code', 'end_date', 'close']
        for col in required_columns:
            if col not in processed_df.columns:
                raise ValueError(f"缺少必要的列：{col}")

        # 日期格式标准化
        processed_df = self._standardize_date(processed_df, 'end_date')
        if 'rept_data_date' in processed_df.columns:
            processed_df = self._standardize_date(processed_df, 'rept_data_date')

        # 日期筛选
        if self.config.start_date and self.config.end_date:
            date_cond = (
                (processed_df['end_date'] >= self.config.start_date) &
                (processed_df['end_date'] <= self.config.end_date)
            )
            processed_df = processed_df[date_cond]

        # 数值类型转换
        numeric_columns = [
            'close', 'volume', 'amount', 'market_value', 'pb', 'pe',
            'divide_rate_ttm', 'month_turnover', 'quarter_turnover', 'annual_turnover',
            'pred_net_profit_yoy', 'total_asset', 'total_liability', 'total_equity',
            'total_income', 'total_income_ttm', 'total_cost', 'total_cost_ttm',
            'net_profit', 'net_profit_ttm', 'income_ps', 'eps',
            'cash', 'ebit', 'ebit_ttm', 'circulate_shares',
            'long_liability_divide_total_equity', 'not_liquid_liability_tot',
            'index_close', 'rf', 'pct_change', 'prev_close',
            # Value/Quality 因子新增字段
            'corp_val_crcp', 'pred_net_profit_ttm', 'pred_eps_std',
            'accr_bs', 'capex', 'cash_flow'
        ]
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

        return processed_df

    def _process_industry_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理行业数据：标准化列名、数据类型转换

        Parameters
        ----------
        df : pd.DataFrame
            原始行业数据

        Returns
        -------
        pd.DataFrame
            处理后的行业数据
        """
        processed_df = df.copy()

        # 列名映射
        column_mapping = {
            'secu_code': 'secu_code',
            'stock_code': 'secu_code',
            'scr_code': 'secu_code',
            'code': 'secu_code',

            'secu_abbr': 'secu_name',
            'stock_name': 'secu_name',

            'company_code': 'company_code',

            'first_industry_code': 'industry_code_l1',
            'sw2021_level1_code': 'industry_code_l1',

            'first_industry_name': 'industry_name_l1',
            'sw2021_level1_name': 'industry_name_l1',

            'second_industry_code': 'industry_code_l2',
            'sw2021_level2_code': 'industry_code_l2',

            'second_industry_name': 'industry_name_l2',
            'sw2021_level2_name': 'industry_name_l2',

            'third_industry_code': 'industry_code_l3',
            'sw2021_level3_code': 'industry_code_l3',

            'third_industry_name': 'industry_name_l3',
            'sw2021_level3_name': 'industry_name_l3',

            'info_pub_date': 'info_pub_date',
            'info_publ_date': 'info_pub_date',

            'insert_time': 'insert_time',
        }

        # 重命名列
        rename_dict = {}
        for col in processed_df.columns:
            col_lower = col.lower()
            if col_lower in column_mapping:
                rename_dict[col] = column_mapping[col_lower]

        if rename_dict:
            processed_df = processed_df.rename(columns=rename_dict)

        # 确保必要的列存在
        required_columns = ['secu_code']
        for col in required_columns:
            if col not in processed_df.columns:
                raise ValueError(f"缺少必要的列：{col}")

        # 日期格式标准化
        if 'info_pub_date' in processed_df.columns:
            processed_df = self._standardize_date(processed_df, 'info_pub_date')

        return processed_df

    def _standardize_date(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        标准化日期格式为 YYYYMMDD 字符串

        Parameters
        ----------
        df : pd.DataFrame
            数据 DataFrame
        date_col : str
            日期列名

        Returns
        -------
        pd.DataFrame
            标准化后的数据
        """
        if date_col not in df.columns:
            return df

        # 尝试多种日期格式转换
        if df[date_col].dtype == 'object':
            # 尝试转换为 datetime
            try:
                df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
            except:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
                except:
                    df[date_col] = pd.to_datetime(df[date_col])

        # 转换为 YYYYMMDD 格式字符串
        df[date_col] = df[date_col].dt.strftime('%Y%m%d')

        return df

    # ==================== 数据合并方法 ====================

    def merge_industry_data(self, on: str = 'secu_code', how: str = 'left') -> pd.DataFrame:
        """
        将行业数据合并到基础数据

        Parameters
        ----------
        on : str
            合并键
        how : str
            合并方式：left, right, inner, outer

        Returns
        -------
        pd.DataFrame
            合并后的数据
        """
        if self.raw_data is None:
            raise ValueError("请先加载基础数据")
        if self.industry_data is None:
            raise ValueError("请先加载行业数据")

        merged_df = pd.merge(
            self.raw_data,
            self.industry_data,
            on=on,
            how=how
        )

        self.raw_data = merged_df
        return merged_df

    # ==================== 数据导出方法 ====================

    def get_raw_data(self) -> pd.DataFrame:
        """获取基础数据"""
        if self.raw_data is None:
            raise ValueError("请先加载数据")
        return self.raw_data.copy()

    def get_industry_data(self) -> pd.DataFrame:
        """获取行业数据"""
        if self.industry_data is None:
            raise ValueError("请先加载行业数据")
        return self.industry_data.copy()

    def save_to_pickle(self, file_path: str, data_type: str = 'raw') -> None:
        """
        保存数据到 pickle 文件

        Parameters
        ----------
        file_path : str
            输出文件路径
        data_type : str
            数据类型：'raw' 或 'industry'
        """
        if data_type == 'raw':
            df = self.get_raw_data()
        elif data_type == 'industry':
            df = self.get_industry_data()
        else:
            raise ValueError(f"未知数据类型：{data_type}")

        df.to_pickle(file_path)
        print(f"数据已保存到：{file_path}")

    def save_to_csv(self, file_path: str, data_type: str = 'raw', index: bool = False) -> None:
        """
        保存数据到 CSV 文件

        Parameters
        ----------
        file_path : str
            输出文件路径
        data_type : str
            数据类型：'raw' 或 'industry'
        index : bool
            是否保存索引
        """
        if data_type == 'raw':
            df = self.get_raw_data()
        elif data_type == 'industry':
            df = self.get_industry_data()
        else:
            raise ValueError(f"未知数据类型：{data_type}")

        df.to_csv(file_path, index=index)
        print(f"数据已保存到：{file_path}")

    # ==================== 辅助方法 ====================

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表（去重）

        Returns
        -------
        pd.DataFrame
            股票列表
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")

        return self.raw_data[['secu_code', 'secu_name']].drop_duplicates()

    def get_trade_dates(self) -> pd.Series:
        """
        获取交易日期列表（去重排序）

        Returns
        -------
        pd.Series
            交易日期列表
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")

        return (
            self.raw_data['end_date']
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

    def get_data_summary(self) -> Dict:
        """
        获取数据摘要信息

        Returns
        -------
        Dict
            数据摘要
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")

        summary = {
            'stock_count': self.raw_data['secu_code'].nunique(),
            'date_count': self.raw_data['end_date'].nunique(),
            'record_count': len(self.raw_data),
            'date_range': (
                self.raw_data['end_date'].min(),
                self.raw_data['end_date'].max()
            ),
            'columns': list(self.raw_data.columns)
        }
        return summary


# ==================== SQL 查询模板 ====================

class BarraSQLTemplates:
    """Barra 数据 SQL 查询模板类"""

    @staticmethod
    def get_raw_data_query(start_date: str, end_date: str) -> str:
        """
        获取 Barra 基础数据 SQL 查询（基于 barra_cne6_raw_data.sql）

        Parameters
        ----------
        start_date : str
            开始日期 YYYYMMDD
        end_date : str
            结束日期 YYYYMMDD

        Returns
        -------
        str
            SQL 查询语句
        """
        return f"""
-- ============================================================================
-- Barra CNE6 原始数据取数 SQL
-- ============================================================================

SELECT
    -- 基础信息
    pf.scr_code                              AS secu_code,
    pf.scr_name                              AS secu_name,
    pf.busi_date                             AS end_date,

    -- 1. 个股收益率
    (pf.tdy_clqn_prc - pf.yest_clqn_prc) / pf.yest_clqn_prc AS pct_change,

    -- 2. 沪深 300 收益率
    hs.tdy_clqn_prc                          AS index_close,
    (hs.tdy_clqn_prc - hs.yest_clqn_prc) / hs.yest_clqn_prc AS index_return,

    -- 3. 总市值
    pf.tot_mval                              AS market_value,

    -- 4. BP 账面市值比 (=1/PB)
    CASE WHEN pf.pb > 0 THEN 1.0 / pf.pb ELSE NULL END AS bp,
    pf.pb                                    AS pb,

    -- 5. 股息率
    pf.divd_rate_ttm                         AS divide_rate_ttm,
    prof.pred_divd_rate                      AS pred_divide_rate,

    -- 6. 换手率
    pf.mth_tnrt                              AS month_turnover,
    pf.quar_tnrt                             AS quarter_turnover,
    pf.ann_tnrt                              AS annual_turnover,

    -- 7. 成交量
    pf.mtch_vol                              AS volume,
    pf.mtch_amt                              AS amount,

    -- 8. 流通股本
    pf.cir_capt                              AS circulate_shares,

    -- 9. 分析师预测利润率
    prof.pred_net_prof_gtrt                  AS pred_net_profit_yoy,
    prof.pred_eps_std                        AS pred_eps_std,
    prof.pred_pe_avg                         AS pred_pe_avg,

    -- 10. 总资产
    fi.tot_ast                               AS total_asset,

    -- 11. 总负债
    fi.liab_tot                              AS total_liability,

    -- 12. 营业总收入
    fi.busi_tot_incm                         AS total_income,
    fi.busi_tot_incm_ttm                     AS total_income_ttm,

    -- 13. 净利润
    fi.net_prof                              AS net_profit,
    fi.net_prof_ttm                          AS net_profit_ttm,

    -- 14. 其他财务指标
    fi.cash                                  AS cash,
    fi.ebit                                  AS ebit,
    fi.ebit_ttm                              AS ebit_ttm

FROM riskdm.dm_inrs_stk_perf_dd pf

-- 关联沪深 300 行情
LEFT JOIN riskdws.dws_var_info_indx_quot_dd hs
    ON hs.scr_code = '000300' AND hs.busi_date = pf.busi_date

-- 关联盈利预测（取最新）
LEFT JOIN riskdm.dm_inrs_stk_prof_cmph_perd_dd prof
    ON prof.scr_code = pf.scr_code
    AND prof.busi_date = (
        SELECT MAX(p2.busi_date) FROM riskdm.dm_inrs_stk_prof_cmph_perd_dd p2
        WHERE p2.scr_code = pf.scr_code AND p2.busi_date <= pf.busi_date
    )

-- 关联财务指标（取最新财报）
LEFT JOIN riskdm.dm_inrs_ins_fin_indx_dd fi
    ON fi.scr_code = pf.scr_code
    AND fi.rept_data_date = (
        SELECT MAX(f2.rept_data_date) FROM riskdm.dm_inrs_ins_fin_indx_dd f2
        WHERE f2.scr_code = pf.scr_code AND f2.rept_data_date <= pf.busi_date
    )

WHERE pf.busi_date >= '{start_date}'
  AND pf.busi_date <= '{end_date}'
  AND pf.tdy_clqn_prc IS NOT NULL
  AND pf.yest_clqn_prc IS NOT NULL
  AND pf.yest_clqn_prc > 0

ORDER BY pf.busi_date DESC, pf.scr_code
;
        """

    @staticmethod
    def get_barra_data(start_date: str, end_date: str) -> str:
        return f"""
        SELECT 
            -- 基础维度：使用 COALESCE 确保在 Outer Join 下日期和代码不为空
            COALESCE(perf.busi_date, prof.busi_date, fin.busi_date, idx.busi_date) AS busi_date,
            COALESCE(perf.scr_code, prof.scr_code, fin.scr_code) AS scr_code,
            
            fin.rept_data_date,
            -- 1. Size Factor (尺寸因子)
            perf.tot_mval AS market_value,             -- 总市值 (用于 LNCAP 计算)
            
            -- 2. Volatility & Momentum Factors (波动率与动量因子)
            perf.plmt AS pct_change,                   -- 个股日收益率
            perf.tdy_clqn_prc AS close,                -- 今日收盘价 (用于 CMRA/PRED_VOEP)
            (idx.tdy_clqn_prc / idx.yest_clqn_prc - 1) AS index_return, -- 指数收益率 (用于 Beta/RSTR 计算)
            
            -- 3. Dividend Yield Factor (红利因子)
            prof.pred_divd_rate,                       -- 预测股息率
            perf.divd_rate_ttm,                        -- 股息率 TTM
            
            -- 4. Value Factor (价值因子)
            CASE WHEN perf.pb > 0 THEN 1.0 / perf.pb ELSE NULL END AS bp,
            fin.net_prof_ttm,                          -- 净利润 TTM (用于 EPTTM)
            fin.ebit_ttm,                              -- 息税前利润 TTM (用于 EBIT_EV)
            fin.corp_val_crcp,                         -- 企业价值 (用于 EBIT_EV 分母)
            prof.pred_pera_avg,                        -- 分析师预测 E/P (预测市盈率倒数)
            
            -- 5. Quality Factor (质量因子)
            fin.liab_tot AS total_liability,           -- 总负债 (用于 DTOA)
            fin.tot_ast AS total_asset,                -- 总资产 (用于 DTOA/AT/GP/ROA 等)
            fin.busi_tot_incm_ttm AS total_income_ttm, -- 营业总收入 TTM (用于 AT/GP/VOS)
            fin.busi_tot_cost_ttm AS total_cost_ttm,   -- 营业总成本 TTM (用于 GP/GPM)
            fin.net_prof_ttm AS net_profit_ttm,        -- 净利润 TTM (用于 GP/GPM)
            fin.net_cash_oper_ttm AS cash_flow_ttm,    -- 经营性现金流 TTM (用于 VOC)
            prof.pred_eps_stde AS pred_eps_std,        -- 预测 EPS 标准差 (用于 PRED_VOEP)
            perf.cir_capt AS circulate_shares,         -- 流通股本 (用于 IG 增长计算)
            fin.net_cash_ivsm_ttm AS capex,            -- 投资性现金流 (代理资本支出用于 CEG)
            -- 注意：accr_bs (应计项目) 在字典中未明确，通常可使用 (净利润 - 经营现金流) 代理
            (fin.net_prof_ttm - fin.net_cash_oper_ttm) AS accr_bs,
            
            -- 6. Growth Factor (成长因子)
            prof.pred_net_prof_gtrt,                   -- 预测净利润增长率 (用于 EGRLF)
            fin.ps_busi_incm AS income_ps,             -- 每股营业收入 (用于 SGRO)
            fin.eps,                                   -- 每股收益 (用于 EGRO)
            
            -- 7. Liquidity Factor (流动性因子)
            perf.mth_tnrt,                             -- 月换手率 (STOM)
            perf.quar_tnrt,                            -- 季换手率 (STOQ)
            perf.ann_tnrt AS ann_tnrt                  -- 年换手率 (STOA)
        
        FROM 
            -- 1. 股票日频业绩表现表 (主驱动表)
            riskdm.dm_inrs_stk_perf_dd perf
        FULL OUTER JOIN 
            -- 2. 盈利预测综合表 (不定期预测)
            riskdm.dm_inrs_stk_prof_cmph_perd_dd prof
            ON perf.busi_date = prof.busi_date AND perf.scr_code = prof.scr_code
        FULL OUTER JOIN 
            -- 3. 机构财务指标表 (季频财务)
            -- 注意：字典显示该表有关联 ins_num，此处需确保财务表已清洗至 scr_code 级别
            riskdm.dm_inrs_ins_fin_indx_dd fin
            ON perf.busi_date = fin.busi_date AND perf.scr_code = fin.scr_code
        FULL OUTER JOIN 
            -- 4. 指数行情表 (用于获取基准收益率，例如 000300.SH)
            (SELECT busi_date, tdy_clqn_prc, yest_clqn_prc FROM riskdws.dws_var_info_indx_quot_dd WHERE scr_code = '000300.SH') idx
            ON perf.busi_date = idx.busi_date
        WHERE perf.busi_date >= '{start_date}'
          AND perf.busi_date <= '{end_date}'
          AND perf.tdy_clqn_prc IS NOT NULL
          AND perf.yest_clqn_prc IS NOT NULL
          AND perf.yest_clqn_prc > 0
        ORDER BY perf.busi_date DESC, perf.scr_code
        """

    @staticmethod
    def get_industry_data_query() -> str:
        """
        获取 A 股最新申万 2021 行业分类 SQL（基于 lc_exgindustry_a_share_latest.sql）

        Returns
        -------
        str
            SQL 查询语句
        """
        return """
-- ============================================================================
-- 获取 A 股最新申万 2021 行业分类
-- 表名：LC_ExgIndustry
-- 标准：Standard=38（申万行业分类 2021 版）
-- ============================================================================

SELECT
    sm.SecuCode                                    AS secu_code,
    sm.SecuAbbr                                    AS secu_name,
    sm.CompanyCode                                 AS company_code,
    lei.FirstIndustryCode                          AS industry_code_l1,
    lei.FirstIndustryName                          AS industry_name_l1,
    lei.SecondIndustryCode                         AS industry_code_l2,
    lei.SecondIndustryName                         AS industry_name_l2,
    lei.ThirdIndustryCode                          AS industry_code_l3,
    lei.ThirdIndustryName                          AS industry_name_l3,
    lei.InfoPublDate                               AS info_pub_date,
    lei.InsertTime                                 AS insert_time
FROM LC_ExgIndustry lei
INNER JOIN SecuMain sm
    ON lei.CompanyCode = sm.CompanyCode
WHERE lei.Standard = 38
  AND lei.IfPerformed = 1
  AND lei.CancelDate IS NULL
  AND sm.SecuCategory = 1
  AND sm.SecuStatus = 1
  AND lei.InfoPublDate = (
      SELECT MAX(lei2.InfoPublDate)
      FROM LC_ExgIndustry lei2
      WHERE lei2.CompanyCode = lei.CompanyCode
        AND lei2.Standard = 38
        AND lei2.IfPerformed = 1
        AND lei2.CancelDate IS NULL
  )
ORDER BY sm.SecuCode
;
        """


def create_connection():
    return pymysql.connections.Connection(host='',
                                          user='',
                                          password='',
                                          database='jydb',
                                          charset='utf-8')


# ==================== 使用示例 ====================
# if __name__ == "__main__":
#     # 示例：如何 BarraRawData 类

#     # 1. 创建配置
#     config = RawDataConfig(
#         start_date='20250101',
#         end_date='20251231'
#     )

#     # 2. 创建取数实例
#     raw_data = BarraRawData(config)

#     # 3. 加载数据（从文件）
#     raw_data.load_raw_data_from_pickle('本地数据.pkl')
#     raw_data.load_industry_data_from_pickle('行业数据.pkl')

#     # 4. 或者从数据库加载（需要数据库连接）
#     conn = create_connection()  # 创建数据库连接
#     raw_query = BarraSQLTemplates.get_raw_data_query('20250101', '20251231')
#     raw_data.load_raw_data_from_sql(raw_query, conn)

#     industry_query = BarraSQLTemplates.get_industry_data_query()
#     raw_data.load_industry_data_from_sql(industry_query, conn)

#     # 5. 合并行业数据
#     raw_data.merge_industry_data()

#     # 6. 查看数据摘要
#     print(raw_data.get_data_summary())

#     # 7. 保存处理后的数据
#     raw_data.save_to_pickle('barra_raw_processed.pkl')

#     print("BarraRawData 模块加载成功！")
