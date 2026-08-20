# -*- coding: utf-8 -*-

from iFinDPy import *
from datetime import datetime
import pandas as pd
import time as _time
import json
from threading import Thread,Lock,Semaphore
import requests

sem = Semaphore(5)  # 此变量用于控制最大并发数

# 登录函数
def thslogindemo():
    # 输入用户的帐号和密码
    thsLogin = THS_iFinDLogin("zssjzc001","Rp5c3uEY")
    print(thsLogin)
    if thsLogin in {0, -201}:
        print('登录成功')
    else:
        print('登录失败')

def datepool_basicdata_demo(code, date):
    # 通过专题报表函数的板块成分报表和基础数据函数，提取全部A股在2025-05-14日的日不复权收盘价
    # query = f"""select scr_num from riskdim.dim_comm_stk_info_df where mkt_code in ('XSHG', 'XSHE') and stk_clas_name = 'A股'
    # and pri_scr_clas_name not in ('已退市且未代办股份转让的股票', '未上市流通的股票') and scr_code = {code}"""
    # code_use = pd.read_sql(query, conn)
    # code_use = codes.replace('XSHG', 'SH').replace('XSHE', 'SZ')
    code_use = '302132.SZ'
    df1 = THS_BD(code_use,'rating_org_num_g',f'{date},180')
    df2 = THS_BD(code_use,'fore_roe_avg',f'{int(str(date)[:4])-1},{date},OC,180')
    df3 = THS_BD(code_use,'fore_roe_avg',f'{int(str(date)[:4])},{date},OC,180')
    df4 = THS_BD(code_use,'fore_roe_avg',f'{int(str(date)[:4])+1},{date},OC,180')
    df5 = THS_BD(code_use,'fore_roe_avg',f'{int(str(date)[:4])+2},{date},OC,180')
    df6 = THS_BD(code_use,'fore_roe_avg',f'{int(str(date)[:4])+3},{date},OC,180')
    if not df6['fore_roe_avg'].iloc[0]:
        result = [df1['rating_org_num_g'].iloc[0], df2['fore_roe_avg'].iloc[0], df3['fore_roe_avg'].iloc[0], df4['fore_roe_avg'].iloc[0]]
    else:
        result = [df1['rating_org_num_g'].iloc[0], df3['fore_roe_avg'].iloc[0], df4['fore_roe_avg'].iloc[0], df5['fore_roe_avg'].iloc[0]]
    return result

def datapool_realtime_demo():
    # 通过板块成分报表和实时行情函数，提取上证50的全部股票的最新价数据,并将其导出为csv文件
    today_str = datetime.today().strftime('%Y%m%d')
    print('today:{}'.format(today_str))
    data_sz50 = THS_DR("p03291","date="+today_str+";blockname=001005260;iv_type=allcontract","p03291_f001:Y,p03291_f002:Y,p03291_f003:Y,p03291_f004:Y")
    if data_sz50.errorcode != 0:
        print('error:{}'.format(data_sz50.errmsg))
    else:
        seccode_sz50_list = data_sz50.data['p03291_f002'].tolist()
        data_result = THS_RQ(seccode_sz50_list,'latest')
        if data_result.errorcode != 0:
            print('error:{}'.format(data_result.errmsg))
        else:
            data_df = data_result.data
            print(data_df)
            data_df.to_csv('realtimedata_{}.csv'.format(today_str))

def iwencai_demo():
    # 演示如何通过不消耗流量的自然语言语句调用常用数据
    print('输出资金流向数据')
    data_wencai_zjlx = THS_WC('主力资金流向', 'stock')
    if data_wencai_zjlx.errorcode != 0:
        print('error:{}'.format(data_wencai_zjlx.errmsg))
    else:
        print(data_wencai_zjlx.data)

    print('输出股性评分数据')
    data_wencai_xny = THS_WC('股性评分', 'stock')
    if data_wencai_xny.errorcode != 0:
        print('error:{}'.format(data_wencai_xny.errmsg))
    else:
        print(data_wencai_xny.data)

def work(codestr,lock,indilist):
    sem.acquire()
    stockdata = THS_HF(codestr, ';'.join(indilist),'','2025-05-14 09:15:00', '2025-05-14 15:30:00','format:json')
    if stockdata.errorcode != 0:
        print('error:{}'.format(stockdata.errmsg))
        sem.release()
    else:
        print(stockdata.data)
        lock.acquire()
        with open('test1.txt', 'a') as f:
            f.write(str(stockdata.data) + '\n')
        lock.release()
        sem.release()

def multiThread_demo():
    # 本函数为通过高频序列函数,演示如何使用多线程加速数据提取的示例,本例中通过将所有A股分100组,最大线程数量sem进行提取
    # 用户可以根据自身场景进行修改
    today_str = datetime.today().strftime('%Y%m%d')
    print('today:{}'.format(today_str))
    data_alla = THS_DR("p03291","date="+today_str+";blockname=001005010;iv_type=allcontract","p03291_f001:Y,p03291_f002:Y,p03291_f003:Y,p03291_f004:Y")
    if data_alla.errorcode != 0:
        print('error:{}'.format(data_alla.errmsg))
    else:
        stock_list = data_alla.data['p03291_f002'].tolist()

    indi_list = ['close', 'high', 'low', 'volume']
    lock = Lock()

    btime = datetime.now()
    l = []
    for eachlist in [stock_list[i:i + int(len(stock_list) / 10)] for i in
                     range(0, len(stock_list), int(len(stock_list) / 10))]:
        nowstr = ','.join(eachlist)
        p = Thread(target=work, args=(nowstr, lock, indi_list))
        l.append(p)

    for p in l:
        p.start()
    for p in l:
        p.join()
    etime = datetime.now()
    print(etime-btime)

pd.options.display.width = 320
pd.options.display.max_columns = None


def reportDownload():
    df = THS_ReportQuery('300033.SZ','beginrDate:2021-08-01;endrDate:2021-08-31;reportType:901','reportDate:Y,thscode:Y,secName:Y,ctime:Y,reportTitle:Y,pdfURL:Y,seq:Y').data
    print(df)
    for i in range(len(df)):
        pdfName = df.iloc[i,4]+str(df.iloc[i,6])+'.pdf'
        pdfURL = df.iloc[i,5]
        r = requests.get(pdfURL)
        with open(pdfName,'wb+') as f:
            f.write(r.content)


def main():
    # 本脚本为数据接口通用场景的实例,可以通过取消注释下列示例函数来观察效果

    # 登录函数
    thslogindemo()
    # 通过专题报表的板块成分函数和基础数据函数，提取全部A股的全部股票在2025-05-14日的日不复权收盘价
    datepool_basicdata_demo(code='600000.SH', date='2026-05-19')
    #通过专题报表的板块成分函数和实时行情函数，提取上证50的全部股票的最新价数据,并将其导出为csv文件
    # datapool_realtime_demo()
    # 演示如何通过不消耗流量的自然语言语句调用常用数据
    # iwencai_demo()
    # 本函数为通过高频序列函数,演示如何使用多线程加速数据提取的示例,本例中通过将所有A股分100组,最大线程数量sem进行提取
    # multiThread_demo()
    # 本函数演示如何使用公告函数提取满足条件的公告，并下载其pdf
    # reportDownload()

if __name__ == '__main__':
    main()















