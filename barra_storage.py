"""
Barra数据存储管理
================
本地数据存储管理类
"""

import os
import pandas as pd
from glob import glob

from app.model.v5.cne6.const import DATA_DIR
from app.utils.DbUtil import DbUtil


class BarraStorage:
    """Barra数据本地存储管理"""

    # DATA_DIR = "cne6/data/"

    def __init__(self):
        self.ensure_dir()

    def ensure_dir(self):
        """确保目录存在"""
        os.makedirs(DATA_DIR, exist_ok=True)

    def get_latest_exposure_date(self) -> str:
        """
        获取因子暴露最新日期

        Returns:
            str: 最新日期，如'20250601'，无文件返回None

        Logic:
            扫描factors_*.feather文件，取最新end日期
        """
        pattern = os.path.join(DATA_DIR, "factors_*.feather")
        files = sorted(glob(pattern))

        if not files:
            return None

        exposure = pd.read_feather(files[-1])
        return exposure['end_date'].max()

    def save_factor_exposure(self, df: pd.DataFrame, start_date: str, end_date: str):
        """
        保存因子暴露结果

        Parameters:
            df: 因子暴露DataFrame
            start_date: 起始日期，如'20250602'
            end_date: 结束日期，如'20250610'
        """
        filename = f"factors_{start_date}_{end_date}.feather"
        filepath = os.path.join(DATA_DIR, filename)
        df.to_feather(filepath)
        print(f"保存因子暴露: {filepath}, 记录数: {len(df)}")

    def load_latest_exposure(self) -> pd.DataFrame:
        """
        加载最新因子暴露数据

        Returns:
            pd.DataFrame: 最新因子暴露数据，无文件返回None
        """
        latest_date = self.get_latest_exposure_date()
        if not latest_date:
            return None

        pattern = os.path.join(DATA_DIR, f"factors_*_{latest_date}.feather")
        files = glob(pattern)
        if not files:
            return None

        return pd.read_feather(files[0])

    def load_all_exposure(self) -> pd.DataFrame:
        """
        加载全量因子暴露数据（合并所有文件）

        Returns:
            pd.DataFrame: 全量因子暴露数据
        """
        pattern = os.path.join(DATA_DIR, "factors_*.feather")
        files = glob(pattern)

        if not files:
            return None

        dfs = [pd.read_feather(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def get_latest_factor_return_date(self) -> str:
        """
        获取因子收益率最新日期

        Returns:
            str: 最新日期，如'20250601'，无文件返回None

        Logic:
            扫描factors_*.feather文件，取最新end日期
        """
        query = """select max(end_date) max_date from barra_beta"""
        max_date = pd.read_sql(query, DbUtil.get_conn('zhijunfund'))
        if max_date.empty:
            return None
        return str(max_date['max_date'].iloc[0])

    def get_latest_pre_beta_vol_date(self) -> str:
        """
        获取因子暴露最新日期

        Returns:
            str: 最新日期，如'20250601'，无文件返回None

        Logic:
            扫描factors_*.feather文件，取最新end日期
        """
        query = """select max(end_date) max_date from barra_beta_pred"""
        max_date = pd.read_sql(query, DbUtil.get_conn('zhijunfund'))
        if max_date.empty:
            return None
        return str(max_date['max_date'].iloc[0])

    def get_latest_residual_vol_date(self) -> str:
        """
        获取因子暴露最新日期

        Returns:
            str: 最新日期，如'20250601'，无文件返回None

        Logic:
            扫描factors_*.feather文件，取最新end日期
        """
        query = """select max(end_date) max_date from barra_residual_pred"""
        max_date = pd.read_sql(query, DbUtil.get_conn('zhijunfund'))
        if max_date.empty:
            return None
        return str(max_date['max_date'].iloc[0])
