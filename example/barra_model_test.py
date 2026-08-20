"""
Barra CNE6 模型测试用例
====================
测试 Barra 因子暴露计算和因子收益率计算
"""
import os
import numpy as np
import pandas as pd
from app.model.v5.cne6.barra_calculator import BarraCNE6Calculator
from app.model.v5.cne6.barra_factor_return_calculator import (
    calculate_constrained_weighted_factor_returns,
    calculate_barra_factor_return,
    factor_return_vol_forecast_vectorized,
    factor_return_correlation_ewma_vectorized,
    residual_volatility_forecast_vectorized,
    residual_volatility_forecast,
    determine_calc_factor_return_range,
    determine_calc_pre_beta_vol_range,
    determine_calc_residual_vol_range
)
from app.model.v5.cne6.example.barra_test import (
    load_real_barra_data,
    load_real_industry_data,
    load_factor_data,
    load_factor_returns,
    load_real_barra_data_test,
    load_residuals
)
from app.model.v5.cne6.const import DATA_DIR


class RealBarraData:
    """实际 Barra 数据测试类 - 包含取数、整合、因子暴露计算、因子收益率计算"""

    factor_df = None
    industry_df = None
    raw_df = None
    calc = None
    factor_returns = None

    @classmethod
    def setup_class(cls):
        """准备实际数据"""
        print("=" * 60)
        print("准备实际 Barra 数据...")
        cls.raw_df = load_real_barra_data()
        cls.industry_df = load_real_industry_data(cls.raw_df)
        print(f"原始数据: {cls.raw_df.shape}")
        print(f"行业数据: {cls.industry_df.shape}")

        # 初始化计算器
        cls.calc = BarraCNE6Calculator()
        cls.calc.data = cls.raw_df.copy()

        # 计算因子暴露
        cls.factor_df = cls.calc.calculate_factors(
            do_winsorize=True,
            do_neutralize=False,
            do_standardize=True
        )

        # 整合因子数据、行业数据和收益率数据
        cls.factor_df = cls.factor_df.merge(cls.industry_df, on='secu_code', how='left').merge(
            cls.raw_df[['secu_code', 'end_date', 'pct_change', 'index_return']], on=['secu_code', 'end_date'], how='inner')
        cls.factor_df['industry'] = cls.factor_df['industry'].fillna('其他')
        cls.factor_df['excess_return'] = cls.factor_df['pct_change'] - cls.factor_df['index_return']

        for date in pd.date_range('2009-12-31', '2026-12-31', freq='Y'):
            date = date.strftime('%Y%m%d')
            if date == pd.to_datetime('2009-12-31'):
                date0 = date
                continue
            print(date0, date)
            factor_df = cls.factor_df[(cls.factor_df['end_date'] > date0) & (cls.factor_df['end_date'] <= date)]
            factor_df.to_feather(f"""{DATA_DIR}/factors_{date0}_{date}.feather""")
            print(f"""{DATA_DIR}/factors_{date0}_{date}.feather""")
            date0 = date

        print(f"因子数据: {cls.factor_df.shape}")

    @classmethod
    def setup_class_test(cls):
        """准备实际数据"""
        print("=" * 60)
        print("准备实际 Barra 数据...")
        start = (pd.to_datetime('today').normalize() - pd.DateOffset(years=6)).strftime('%Y%m%d')
        # start = '20100101'
        cls.raw_df = load_real_barra_data_test(start)
        cls.industry_df = load_real_industry_data(cls.raw_df)
        print(f"原始数据: {cls.raw_df.shape}")
        print(f"行业数据: {cls.industry_df.shape}")

        # 初始化计算器
        cls.calc = BarraCNE6Calculator()
        cls.calc.data = cls.raw_df.copy()

        # 计算因子暴露
        cls.factor_df = cls.calc.calculate_factors(
            do_winsorize=True,
            do_neutralize=False,
            do_standardize=True
        )
        if cls.factor_df is None:
            return

        # 整合因子数据、行业数据和收益率数据
        cls.factor_df = cls.factor_df.merge(cls.industry_df, on='secu_code', how='left').merge(
            cls.raw_df[['secu_code', 'end_date', 'pct_change', 'index_return']], on=['secu_code', 'end_date'], how='inner')
        cls.factor_df['industry'] = cls.factor_df['industry'].fillna('其他')
        cls.factor_df['excess_return'] = cls.factor_df['pct_change'] - cls.factor_df['index_return']
        start_date, end_date = cls.factor_df['end_date'].min(), cls.factor_df['end_date'].max()
        file_start = pd.to_datetime(start_date) - pd.offsets.YearEnd()
        file_end = pd.to_datetime(end_date) - pd.DateOffset(days=1) + pd.offsets.YearEnd()

        for date in pd.date_range(file_start, file_end, freq='Y'):
            date = date.strftime('%Y%m%d')
            if date == file_start.strftime('%Y%m%d'):
                date0 = date
                continue
            print(date0, date)
            if os.path.exists(f"""{DATA_DIR}/factors_{date0}_{date}.feather"""):
                last = pd.read_feather(f"""{DATA_DIR}/factors_{date0}_{date}.feather""")
            else:
                last = pd.DataFrame()
            factor_df = cls.factor_df[(cls.factor_df['end_date'] > date0) & (cls.factor_df['end_date'] <= date)]
            factor_df = pd.concat([last, factor_df]).reset_index(drop=True)
            factor_df.to_feather(f"""{DATA_DIR}/factors_{date0}_{date}.feather""")
            print(f"""{DATA_DIR}/factors_{date0}_{date}.feather""")
            date0 = date

        print(f"因子数据: {cls.factor_df.shape}")

    def calculate_all_factor_returns(self):
        """测试全量因子收益率计算"""
        print("\n[测试 18] 测试全量因子收益率计算")
        style_factors = ['size', 'volatility', 'momentum', 'value', 'dividend_yield', 'quality', 'growth', 'liquidity']
        start = (pd.to_datetime('today').normalize() - pd.DateOffset(years=1)).strftime('%Y%m%d')
        self.factor_df = load_factor_data(start)
        last, _, _, cal_type = determine_calc_factor_return_range(self.factor_df)
        if not cal_type:
            return
        if last:
            self.factor_df = self.factor_df[self.factor_df['end_date'] >= last]
        calculate_barra_factor_return(
            self.factor_df,
            style_factors_to_use=style_factors,
            industry_col='industry',
            size_col='size',
            excess_return_col='pct_change'
        )
        print(r"[PASS] 全量因子收益率计算完成")

    def calculate_factor_prediction(self):
        """测试全量因子收益率计算"""
        print("\n[测试 19] 测试全量因子波动率预测")
        self.factor_returns = load_factor_returns()

        start, _, cal_type = determine_calc_pre_beta_vol_range(self.factor_returns)
        if not cal_type:
            return
        if start:
            dates = self.factor_returns[self.factor_returns['end_date'] >= start]['end_date'].tolist()
        else:
            dates = self.factor_returns['end_date'].tolist()
        factor_return_vol_forecast_vectorized(
            self.factor_returns,
            dates,
            n_jobs=4)
        print(r"[PASS] 全量因子波动率预测计算完成")

    def calculate_factor_corr(self):
        """测试全量因子收益率计算"""
        print("\n[测试 20] 测试因子相关系数")
        self.factor_returns = load_factor_returns()
        start = self.factor_returns['end_date'].tolist()[-90]
        factor_return_correlation_ewma_vectorized(
            self.factor_returns,
            start)
        print(r"[PASS] 全量因子相关系数计算完成")

    def calculate_residual_prediction(self):
        """测试全量因子收益率计算"""
        print("\n[测试 21] 测试全量残差波动率预测")
        self.residual = load_residuals()
        start = self.residual.index.tolist()[-90]
        # residual_volatility_forecast_vectorized(
        #     self.residual,
        #     self.residual.index,
        #     n_jobs=4)
        residual_volatility_forecast(
            self.residual,
            start)
        print(r"[PASS] 全量残差波动率预测计算完成")


def run_real_data():
    """运行实际数据测试"""
    print("\n" + "=" * 60)
    print("第三部分：实际 Barra 数据完整测试")
    print("=" * 60)
    RealBarraData.setup_class_test()
    # RealBarraData.setup_class()
    real_tests = RealBarraData()
    real_tests.calculate_all_factor_returns()
    real_tests.calculate_factor_prediction()
    real_tests.calculate_factor_corr()
    real_tests.calculate_residual_prediction()

    print("\n" + "=" * 60)
    print("实际数据测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    # run_all()
    run_real_data()
