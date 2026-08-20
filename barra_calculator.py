"""
Barra CNE6 计算器
==================
主程序入口，整合数据加载、因子计算、结果保存
"""

import pandas as pd
from typing import Optional

from app.model.v5.cne6.conf.configs import PreprocessConfig, FactorConfig
from app.model.v5.cne6.barra_base_functions import BarraBaseFunctionsUse
from app.model.v5.cne6.barra_factors import BarraFactorFactory
from app.model.v5.cne6.barra_storage import BarraStorage


class BarraCNE6Calculator:
    """Barra 统一计算器"""

    def __init__(self, storage: Optional[BarraStorage] = None):
        self.storage = storage or BarraStorage()
        self.preprocess_cfg = PreprocessConfig()
        self.factor_cfg = FactorConfig()
        self.factory = BarraFactorFactory(self.preprocess_cfg, self.factor_cfg)
        self.data: Optional[pd.DataFrame] = None
        self.factor_data: Optional[pd.DataFrame] = None

    def _determine_calc_range(self) -> tuple:
        """确定计算时段"""
        dates = self.data['end_date'].drop_duplicates().sort_values().tolist()
        source_start_date, source_end_date = dates[0], dates[-1]
        latest_exposure_date = self.storage.get_latest_exposure_date()

        if latest_exposure_date:
            if latest_exposure_date >= dates[-1]:
                print(f"因子暴露数据已是最新: latest={latest_exposure_date}, source_end={source_end_date}")
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

    def calculate_factors(self,
                          do_winsorize=True,
                          do_neutralize=True,
                          do_standardize=True
                          ) -> pd.DataFrame:
        """计算因子"""
        if self.data is None:
            raise ValueError("请先加载数据")

        start_date, _, cal_type = self._determine_calc_range()
        if not cal_type:
            return
        self.factor_data = self.factory.calculate_all_factors(self.data, start_date)

        # 统一预处理
        if do_winsorize or do_neutralize or do_standardize:
            base = BarraBaseFunctionsUse(self.factor_data, self.preprocess_cfg)
            base.factor_cols = self.factory.get_factor_list()
            self.factor_data = base.preprocess(
                do_winsorize=do_winsorize,
                do_neutralize=do_neutralize,
                do_standardize=do_standardize
            )

        print(f"因子计算完成: {len(self.factor_data)} 条记录")
        return self.factor_data

    def save_results(self):
        """保存因子暴露结果"""
        if self.factor_data is None:
            return

        self.storage.save_factor_exposure(
            self.factor_data,
            self.calc_start_date,
            self.calc_end_date
        )