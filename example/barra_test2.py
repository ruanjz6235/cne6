from glob import glob
import pandas as pd

lst = glob(r'D:\claude\Barra\cne6\data\barra_test\dm_inrs_ins_fin_indx_dd_*.feather')
lst1 = glob(r'D:\claude\Barra\cne6\data\barra_test\dm_inrs_stk_prof_cmph_perd_dd_*.feather')

d_s = pd.read_feather(r'D:\claude\Barra\cne6\data\barra_test\dws_var_info_indx_quot_dd.feather')

a_s = []
for file in lst:
    # print(file[61: 69])
    a_s.append(pd.read_feather(file))
a_s = pd.concat(a_s)
a_s = a_s[(a_s['secu_code'].str.startswith('6')) | (a_s['secu_code'].str.startswith('3')) | (a_s['secu_code'].str.startswith('0'))]
print('success')

b_s = pd.read_feather(r'D:\claude\Barra\cne6\data\barra_test\dm_inrs_stk_prof_cmph_perd_dd.feather')
b_s = b_s[(b_s['secu_code'].str.startswith('6')) | (b_s['secu_code'].str.startswith('3')) | (b_s['secu_code'].str.startswith('0'))]
print('success')

c_s = b_s.merge(a_s, on=['secu_code', 'end_date'], how='outer')
c_s = c_s[~c_s['corp_val_crcp'].isna()]

c_s.columns = ['end_date', 'secu_code', 'pred_divd_rate_avg', 'pred_pera_avg',
               'pred_eps_std', 'pred_net_prof_gtrt', 'rela_date', 'rept_data_date',
               'ins_num', 'net_prof_ttm', 'ebit_ttm', 'total_liability', 'total_asset',
               'total_income_ttm', 'total_cost_ttm', 'cash_flow_ttm', 'capex',
               'accr_bs', 'income_ps', 'eps', 'market_value', 'pct_change', 'close',
               'divd_rate_ttm', 'bp', 'corp_val_crcp', 'circulate_shares', 'mth_tnrt',
               'quar_tnrt', 'ann_tnrt']
d_s.columns = ['end_date', 'close', 'index_return']
print(1)


