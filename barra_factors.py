"""
Barra CNE6 因子类
==================
包含 8 大风格因子计算类
"""

import numpy as np
import pandas as pd
from typing import Optional, List

from app.model.v5.cne6.conf.configs import PreprocessConfig, FactorConfig
from app.model.v5.cne6.barra_base_functions import BarraBaseFunctions


base_cols = ['secu_code', 'end_date']


# done
class SizeFactor(BarraBaseFunctions):
    """Size 尺寸因子 - ln(市值)"""

    def __init__(self, df: pd.DataFrame = None, config: Optional[PreprocessConfig] = None):
        self.raw_cols = ['market_value']
        super().__init__(df, config)
        self.factor_name = 'size'
        self.factor_cols = ['LNCAP', 'MIDCAP']

    def LNCAP(self):
        self.data['LNCAP'] = np.log(self.data['market_value'])

    def MIDCAP(self):
        self.data['MIDCAP'] = self.data['LNCAP'] ** 3 - self.data['LNCAP']

    def calculate(self) -> pd.DataFrame:
        # todo: 退市后的行情数据直接删除整个证券，还是保留历史数据。保留历史数据的有000024，000038，000418；删除整个证券的有【600462】。如下是测试代码
        # for i in mv_unstack.columns:
        #     index = list(mv_unstack.index).index(mv_unstack[~mv_unstack[i].isna()].index[0])
        #     nan_count = len(mv_unstack[mv_unstack[i].isna()])
        #     if nan_count > index:
        #         print(i)
        # todo: 停牌是按照ffill的方式填充(000002)，还是不填充(000155)
        self.LNCAP()
        self.MIDCAP()
        self.preprocess(self.factor_cols)
        self.data[self.factor_name] = self.data[self.factor_cols].mean(axis=1, skipna=True)
        return self.data[base_cols + [self.factor_name]].set_index(base_cols)


# done
class VolatilityFactor(BarraBaseFunctions):
    """Beta 贝塔因子 - 滚动加权回归"""

    def __init__(self, df: pd.DataFrame = None, preprocess_cfg=None, factor_cfg=None):
        self.raw_cols = ['pct_change', 'index_return']
        super().__init__(df, preprocess_cfg)
        self.factor_config = factor_cfg or FactorConfig()
        self.factor_name = 'volatility'
        self.factor_cols = ['beta', 'hsigma', 'DSTD', 'CMRA']
        # todo: pct_change同market_value的问题一样
        self.weighted_beta_vectorize(self.factor_config, weight_method='exp')

    def BETA(self):
        beta = self.result.beta.stack().reset_index()
        beta.columns = ['end_date', 'secu_code', 'beta']
        self.data = self.data.merge(beta, on=base_cols, how='left')

    # Residual_Volatility
    def HSIGMA(self):
        hsigma = self.result.residual_std.stack().reset_index()
        hsigma.columns = ['end_date', 'secu_code', 'hsigma']
        self.data = self.data.merge(hsigma, on=base_cols, how='left')

    def HALPHA(self) -> pd.DataFrame:
        return self.result.alpha.stack().reset_index()

    def DSTD(self):
        config = FactorConfig(window=252, beta_half_life=42)
        self.data['DSTD'] = self.ewm(raw_col='pct_change', output_type='std', config=config)

    def CMRA(self):
        self.data['log_return'] = np.log(self.data['pct_change']+1)
        rs = []
        for i in range(12):
            rs.append(self.data.groupby('secu_code').rolling(21*(i+1))['log_return'].sum().reset_index(level=0)['log_return'])
        rs = pd.concat(rs, axis=1).reset_index(drop=True)
        self.data['CMRA'] = rs.max(axis=1, skipna=False) - rs.min(axis=1, skipna=False)

    def calculate(self) -> pd.DataFrame:
        self.BETA()
        self.HSIGMA()
        self.DSTD()
        self.CMRA()
        self.preprocess(self.factor_cols)
        self.data[self.factor_name] = self.data[self.factor_cols].mean(axis=1, skipna=True)
        return self.data[base_cols + [self.factor_name]].set_index(base_cols)


class DividendYieldFactor(BarraBaseFunctions):
    """DividendYield 红利因子"""

    def __init__(self, df: pd.DataFrame = None, config: Optional[PreprocessConfig] = None):
        self.raw_cols = ['pred_divd_rate', 'divd_rate_ttm']
        super().__init__(df, config)
        self.factor_name = 'dividend_yield'
        self.factor_cols = ['DY']

    def DY(self):
        # pred_divd_rate字段存在大量为0的数值，到底是预测值为0，还是没有预测值，如000001的pred_divd_rate字段为0时是预测值为0，而000006的pred_divd_rate字段为0时是没有预测值填充成了0
        # 000001的divd_rate_ttm字段为0（20100104~20121019），是没有数据，还是数值为0，000008缺失数据
        
        for i in self.raw_cols:
            self.data[i+'_tmp'] = self.data[i]
            self.data.loc[self.data[i] == 0, i+'_tmp'] = np.nan
        self.data[self.factor_name] = self.data[['pred_divd_rate_tmp', 'divd_rate_ttm_tmp']].mean(axis=1).fillna(0)

    def calculate(self) -> pd.DataFrame:
        self.DY()
        self.preprocess([self.factor_name])
        return self.data[base_cols + [self.factor_name]].set_index(base_cols)


# done
class MomentumFactor(BarraBaseFunctions):
    """Momentum 动量因子 - RSTR + HALPHA"""

    def __init__(self, df: pd.DataFrame = None, preprocess_cfg=None, factor_cfg=None):
        self.raw_cols = ['pct_change', 'index_return']
        super().__init__(df, preprocess_cfg)
        self.factor_config = factor_cfg or FactorConfig()
        self.factor_name = 'momentum'
        self.factor_cols = ['RSTR', 'halpha', 'STREV']

    def STREV(self):
        config = FactorConfig(window=21, beta_half_life=5)
        self.data['STREV'] = self.ewm(raw_col='pct_change', config=config)

    def SEASON(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("请先设置数据")
        df = self.data.copy()
        return df[['secu_code', 'end_date', 'SEASON']]

    def INDMOM(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("请先设置数据")
        df = self.data.copy()
        return df[['secu_code', 'end_date', 'INDMOM']]

    def RSTR(self):
        self.data['excess_return'] = self.data['pct_change'] - self.data['index_return']
        config = FactorConfig(window=252, beta_half_life=126)
        self.data['RSTR_tmp'] = self.ewm(raw_col='excess_return', config=config)
        self.data['RSTR'] = self.data.groupby('secu_code')['RSTR_tmp'].rolling(11).mean().reset_index(level=0)['RSTR_tmp']

    def HALPHA(self, vol_factor: VolatilityFactor = None):
        halpha = vol_factor.result.alpha.stack().reset_index()
        halpha.columns = ['end_date', 'secu_code', 'halpha']
        halpha = halpha[~halpha['halpha'].isna()]
        self.data = self.data.merge(halpha, on=base_cols, how='left')

    def calculate(self, vol_factor: VolatilityFactor = None) -> pd.DataFrame:
        self.RSTR()
        self.HALPHA(vol_factor)
        self.STREV()
        self.preprocess(self.factor_cols)
        self.data['sub_momentum'] = self.data[['RSTR', 'halpha']].mean(axis=1, skipna=True)
        self.data[self.factor_name] = self.data[['sub_momentum', 'STREV']].mean(axis=1, skipna=True)
        return self.data[base_cols + [self.factor_name]].set_index(base_cols)


class ValueFactor(BarraBaseFunctions):
    def __init__(self, df: pd.DataFrame = None, preprocess_cfg=None, factor_cfg=None):
        self.raw_cols = ['bp', 'net_profit_ttm', 'market_value', 'ebit_ttm', 'corp_val_crcp', 'pct_change', 'index_return', 'pred_pera_avg']
        super().__init__(df, preprocess_cfg)
        self.factor_cfg = factor_cfg or FactorConfig()
        self.factor_name = 'value'
        # bp指标无科创板数据，缺失数据的股票还有
        # ['600608', '600603', '000509', '000504', '600421', '002306', '600338', '600234', '600155', '600617', '000958', '000555', '000017', '000035', '600678', '000908', '600579', '600198', '600444', '000953', '002188', '600892', '600751', '600691', '600771', '600988', '000820', '600877', '600722', '000557', '300093', '000692', '000927', '002076', '002356', '600301', '300268', '600581', '600179', '000615', '600885', '600706', '000056', '300338', '000520', '002168', '600654', '300159', '002713', '600217', '600793', '000410', '002072', '600734', '600715', '000892', '600381', '600757', ...]
        # ebit_ttm == 0，['000001', '002142', '600016', '600015', '600036', '600109', '600369', '600030', '600000', '601099', '600999', '601009', '601398', '601318', '601998', '601988', '601939', '601628', '601601', '601788', '601166', '601169', '601328', '000728', '000686', '000563', '000783', '601688', '000776', '601288', '601818', '601377', '002500', '600837', '601901', '000750', '601555', '601336', '002673', '002736', '000166', '601198', '600958', '601211', '002797', '600919', '601997', '002807', '600908', '601128', '600926', '000627', '601229', '603323', '600909', '601375', '601881', '002839', '601878', ...]
        # net_profit_ttm == 0无科创板数据
        self.factor_cols = ['bp', 'EPTTM', 'EBIT_EV', 'pred_pera_avg', 'LTRSTR', 'LTHALPHA']
        # self.factor_cols = ['bp', 'EPTTM', 'EBIT_EV', 'pred_pera_avg', 'LTRSTR']

    def EPTTM(self):
        self.data['EPTTM'] = self.data['net_profit_ttm'] / self.data['market_value']

    def EBIT_EV(self):
        """Earning_Yield"""
        self.data['EBIT_EV'] = self.data['ebit_ttm'] / self.data['corp_val_crcp']

    def LTRSTR(self):
        self.data['excess_return'] = self.data['pct_change'] - self.data['index_return']
        config = FactorConfig(window=1040, beta_half_life=260)
        self.data['LTRSTR_tmp'] = self.ewm(raw_col='excess_return', config=config)
        self.data['LTRSTR'] = self.data.groupby('secu_code')['LTRSTR_tmp'].rolling(11).mean().reset_index(level=0)['LTRSTR_tmp']

    def LTHALPHA(self):
        """Earning_Yield"""
        config = FactorConfig(window=1040, beta_half_life=252)
        self.weighted_beta_vectorize(config)
        df_new = self.result.alpha.stack().reset_index()
        df_new.columns = ['end_date', 'secu_code', 'LTHALPHA']
        self.data = self.data.merge(df_new, on=base_cols, how='left')

    def calculate(self):
        self.EPTTM()
        self.EBIT_EV()
        self.LTRSTR()
        self.LTHALPHA()
        self.preprocess(self.factor_cols)
        self.data[self.factor_name] = self.data[self.factor_cols].mean(axis=1, skipna=True)
        return self.data[base_cols + [self.factor_name]].set_index(base_cols)


class QualityFactor(BarraBaseFunctions):
    """
    Quality 质量因子 - 分层合成

    子维度:
    1. Leverage (杠杆): DTOA
    2. Profitability (盈利能力): ATO, GP, GPM, ROA
    3. Earnings Variability (盈利波动): VOS, VOE, VOC, PRED_VOEP
    4. Earnings Quality (盈利质量): ABS
    5. Investment Quality (投资质量): TAG, IG, CEG

    合成方式:
    - Leverage = DTOA
    - Profitability = (ATO + GP + GPM + ROA) 等权平均
    - Earnings Variability = (VOS + VOE + VOC + PRED_VOEP) 等权平均
    - Earnings Quality = ABS
    - Investment Quality = (TAG + IG + CEG) 等权平均
    - Quality = (Leverage + Profitability + EV + EQ + IQ) 等权平均
    """

    def __init__(self, df: pd.DataFrame = None, config: Optional[PreprocessConfig] = None):
        self.raw_cols = ['total_liability', 'total_asset', 'total_income_ttm', 'total_cost_ttm', 'net_profit_ttm',
                         'cash_flow_ttm', 'pred_eps_std', 'close', 'accr_bs', 'circulate_shares', 'capex', 'rept_data_date']
        super().__init__(df, config)
        self.factor_name = 'quality'
        self.factor_cols = ['DTOA', 'AT', 'GP', 'ROA', 'VOS', 'VOE', 'VOC', 'PRED_VOEP', 'ABS', 'TAG', 'IG', 'CEG']

    # ========== Leverage 子因子 ==========
    def DTOA(self):
        """资产负债率 = 总负债 / 总资产"""
        self.data['DTOA'] = self.data['total_liability'] / self.data['total_asset']

    # ========== Profitability 子因子 ==========
    def AT(self):
        """资产周转率 = 营业收入(TTM) / 总资产"""
        # todo: 未上市的股票要先筛除，否则会出现total_income_ttm不等于0而total_asset等于0
        self.data['AT'] = self.data['total_income_ttm'] / self.data['total_asset']

    def GP(self):
        """总盈利能力 = (营业收入 - 营业成本) / 总资产"""
        # todo: 存在营业收入等于0而营业成本大于0，营业收入大于0而营业成本等于0，营业收入或营业成本小于0
        self.data['GP'] = (self.data['total_income_ttm'] - self.data['total_cost_ttm']) / self.data['total_asset']

    def GPM(self):
        """毛利率 = (营业收入 - 营业成本) / 营业收入"""
        self.data['GPM'] = (self.data['total_income_ttm'] - self.data['total_cost_ttm']) / self.data['total_income_ttm']

    def ROA(self):
        """资产收益率 = 净利润(TTM) / 总资产"""
        self.data['ROA'] = self.data['net_profit_ttm'] / self.data['total_asset']

    # ========== Earnings Variability 子因子 ==========
    def VOS(self):
        """营业收入波动性 = 5年营业收入标准差 / 均值"""
        # todo: total_income_ttm指标存在缺失值，如000001, 689009所有的数值都为0。
        self.calc_5y_volatility('total_income_ttm', 'VOS')

    def VOE(self):
        """盈利波动性 = 5年净利润标准差 / 均值"""
        self.calc_5y_volatility('net_profit_ttm', 'VOE')

    def VOC(self):
        """现金流波动性 = 5年经营现金流标准差 / 均值"""
        self.calc_5y_volatility('cash_flow_ttm', 'VOC')

    def PRED_VOEP(self):
        """分析师预测盈利波动性 = 预期EPS标准差 / 当前价格"""
        self.data['PRED_VOEP'] = self.data['pred_eps_std'] / self.data['close']

    # ========== Earnings Quality 子因子 ==========
    def ABS(self):
        """
        应计项目 (ABS) = -ACCR_BS / 总资产

        做多应计项目较低的公司
        """
        self.data['ABS'] = -self.data['accr_bs'] / self.data['total_asset']

    # ========== Investment Quality 子因子 ==========
    def TAG(self):
        # todo: total_asset字段有不少股票，如000004缺失数据，只有20091231核20161231两条数据(注：000004所有的字段都只有这两条年末的数据)
        # 用以下公式检验
        # annual_data.groupby('secu_code')['rept_data_date'].count()[annual_data.groupby('secu_code')['rept_data_date'].count() < (annual_data.groupby('secu_code')['rept_data_date'].max().str[:4].astype(int) - annual_data.groupby('secu_code')['rept_data_date'].min().str[:4].astype(int) + 1)]
        """总资产增长率 = -5年总资产回归斜率 / 均值"""
        self.calc_5y_regression_slope('total_asset', 'TAG')

    def IG(self):
        """发行增长 = -5年流通股回归斜率 / 均值"""
        self.calc_5y_regression_slope('circulate_shares', 'IG')

    def CEG(self):
        """资本支出增长 = -5年资本支出回归斜率 / 均值"""
        self.calc_5y_regression_slope('capex', 'CEG')

    def calculate(self) -> pd.DataFrame:
        """计算Quality因子 - 分层合成"""
        # ========== 第一层: 子因子计算 ==========
        # Leverage
        self.DTOA()
        # Profitability
        self.AT()
        self.GP()
        self.ROA()

        # Earnings Variability
        self.VOS()
        self.VOE()
        self.VOC()
        self.PRED_VOEP()

        # Earnings Quality
        self.ABS()

        # Investment Quality
        self.TAG()
        self.IG()
        self.CEG()
        self.preprocess(self.factor_cols)

        # ========== 第二层: 子维度合成 ==========
        # Leverage = DTOA (单独因子)
        # Profitability = (ATO + GP + GPM + ROA) 等权平均
        # Earnings Variability = (VOS + VOE + VOC + PRED_VOEP) 等权平均
        # Earnings Quality = ABS (单独因子)
        # Investment Quality = (TAG + IG + CEG) 等权平均
        # ========== 第三层: 主因子合成 ==========
        # Quality = (Leverage + Profitability + EV + EQ + IQ) 等权平均
        # quality_cols = [c for c in ['leverage_factor', 'profitability_factor',
        #                             'earnings_variability', 'earnings_quality',
        #                             'investment_quality'] if c in df.columns]
        self.data[self.factor_name] = self.data[self.factor_cols].mean(axis=1, skipna=True)
        return self.data[base_cols + [self.factor_name]].set_index(base_cols)


class GrowthFactor(BarraBaseFunctions):
    """Growth 成长因子 - EGRLF/EGRO/SGRO"""

    def __init__(self, df: pd.DataFrame = None, config: Optional[PreprocessConfig] = None):
        self.raw_cols = ['pred_net_prof_gtrt', 'income_ps', 'eps', 'rept_data_date']
        super().__init__(df, config)
        self.factor_name = 'growth'
        self.factor_cols = ['EGRLF', 'SGRO', 'EGRO']

    def EGRLF(self):
        self.data = self.data.rename(columns={'pred_net_prof_gtrt': 'EGRLF'})

    def SGRO(self):
        self.calc_5y_regression_slope('income_ps', 'SGRO')

    def EGRO(self):
        self.calc_5y_regression_slope('eps', 'EGRO')

    def calculate(self) -> pd.DataFrame:
        self.EGRLF()
        self.EGRO()
        self.SGRO()
        self.preprocess(self.factor_cols)
        self.data[self.factor_name] = self.data[self.factor_cols].mean(axis=1, skipna=True)
        return self.data[base_cols + [self.factor_name]].set_index(base_cols)


class LiquidityFactor(BarraBaseFunctions):
    """Liquidity 流动性因子 - STOM/STOQ/STOA"""

    def __init__(self, df: pd.DataFrame = None, config: Optional[PreprocessConfig] = None):
        self.raw_cols = ['mth_tnrt', 'quar_tnrt', 'ann_tnrt']
        super().__init__(df, config)
        self.factor_name = 'liquidity'
        self.factor_cols = ['stom', 'stoq', 'stoa']

    def calculate(self) -> pd.DataFrame:
        self.data['stom'] = np.log(self.data['mth_tnrt']+1)
        self.data['stoq'] = np.log(self.data['quar_tnrt']+1)
        self.data['stoa'] = np.log(self.data['ann_tnrt']+1)
        self.preprocess(self.factor_cols)
        self.data[self.factor_name] = self.data[self.factor_cols].mean(axis=1, skipna=True)
        return self.data[base_cols + [self.factor_name]].set_index(base_cols)


class BarraFactorFactory:
    """因子工厂 - 统一管理所有因子计算"""
    WINDOWS = {
        'size': 0,            # 无历史需求
        'liquidity': 0,       # 无历史需求
        'dividend_yield': 0,  # 无历史需求
        'momentum': 504,      # RSTR(252) + halpha(504)，取最大
        'volatility': 504,    # beta(504)最大
        'value': 1040,        # LTRSTR(1040) + LTHALPHA(1040)
        'quality': 1260,      # 5年≈1260交易日
        'growth': 1260        # 5年≈1260交易日
    }

    def __init__(self, preprocess_cfg=None, factor_cfg=None, factors=None):
        self.preprocess_cfg = preprocess_cfg or PreprocessConfig()
        self.factor_cfg = factor_cfg or FactorConfig()
        self.factors = factors

    def calculate_all_factors(self, df: pd.DataFrame, start) -> pd.DataFrame:
        """计算所有因子"""
        dates = df['end_date'].drop_duplicates().sort_values().tolist()
        if start:
            dfs = {k: df[df['end_date'] >= dates[dates.index(start)-v]] for k, v in self.WINDOWS.items()}
        else:
            dfs = {k: df for k, _ in self.WINDOWS.items()}
        print('开始计算因子数据')
        self.factors = {
            'size': SizeFactor(dfs['size'], self.preprocess_cfg),
            'momentum': MomentumFactor(dfs['momentum'], self.preprocess_cfg, self.factor_cfg),
            'dividend_yield': DividendYieldFactor(dfs['dividend_yield']),
            'value': ValueFactor(dfs['value'], self.preprocess_cfg, self.factor_cfg),
            'quality': QualityFactor(dfs['quality'], self.preprocess_cfg),
            'volatility': VolatilityFactor(dfs['volatility'], self.preprocess_cfg, self.factor_cfg),
            'growth': GrowthFactor(dfs['growth'], self.preprocess_cfg),
            'liquidity': LiquidityFactor(dfs['liquidity'], self.preprocess_cfg),
        }

        result = []
        vol_factor = self.factors['volatility']
        for name, calc in self.factors.items():
            print(f"计算 {name} 因子...")
            if name == 'momentum':
                df_i = calc.calculate(vol_factor).reset_index()
            else:
                df_i = calc.calculate().reset_index()
            if start:
                df_i = df_i[df_i['end_date'] >= start]
            result.append(df_i.set_index(['secu_code', 'end_date']))
        result = pd.concat(result, axis=1)
        return result.reset_index()

    def calculate_single_factor(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """计算单个因子"""
        if name not in self.factors:
            raise ValueError(f"未知因子：{name}")
        calc = self.factors[name]
        calc.set_data(df.copy())
        return calc.calculate()

    def get_factor_list(self) -> List[str]:
        return list(self.factors.keys())
