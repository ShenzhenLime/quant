import tushare as ts
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timedelta
from tqdm import tqdm

from joblib import Parallel, delayed
import os
from dotenv import load_dotenv
from quant_infra import db_utils
from quant_infra.const import *
from pathlib import Path

# Tushare 配置：优先从仓库根目录 .env 读取 TS_TOKEN。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / '.env')
token = os.getenv('TS_TOKEN')

def _get_pro_client():
    if not token:
        raise RuntimeError('未找到 TS_TOKEN，请在项目根目录的 .env 文件中配置后再运行。')
    # 显式传入 token，避免 tushare 在导入阶段进行 token 连接
    return ts.pro_api(token)

# ====================  数据获取函数  ====================
def get_ins(index_code):
    """获取指数成分股并保存，tushare接口只能获取最近一个月的成分股数据，所以这里自动计算上个月的日期范围"""
    today = datetime.now()
    # 这个月的第一天减去一天，就是上个月的最后一天
    last_day_of_last_month = today.replace(day=1) - timedelta(days=1)
    first_day_of_last_month = last_day_of_last_month.replace(day=1)
    
    start = first_day_of_last_month.strftime('%Y%m%d')
    end = last_day_of_last_month.strftime('%Y%m%d')
    
    try:
        pro = _get_pro_client()
        ins = pro.index_weight(index_code=index_code, start_date=start, end_date=end)
        if ins is None or ins.empty:
            print(f"警告: 未获取到指数 {index_code} 的成分股数据")
            return set()
        
    ## 一些内存较小的表，用csv存储
        Path(BASIC_INFO_PATH).mkdir(parents=True, exist_ok=True)
        ins['con_code'].drop_duplicates().to_csv(f'{BASIC_INFO_PATH}/{index_code}_ins.csv', index=False)

        ins_codes = ins['con_code'].unique().tolist()
        return set(ins_codes)
    except Exception as e:
        # 使用 from e 可以保留原始的错误轨迹 (Traceback)
        raise RuntimeError(f"获取指数成分股失败") from e

def get_trade(start_date, end_date):
    """获取交易日历"""
    pro = _get_pro_client()
    df = pro.trade_cal(exchange='SSE', is_open='1', start_date=start_date, end_date=end_date, fields='cal_date')
    df.to_csv(f'{BASIC_INFO_PATH}/trade_day.csv', index=False)
    return df

def fetch_bar_by_single_date(date):  ## 增加重试机制，避免偶尔的网络问题导致数据缺失
    for i in range(4):  # 最多重试4次
        try:
            pro = _get_pro_client()
            df = pro.daily(trade_date=date)
            if df is not None:
                df = df[df['pct_chg'].abs() < 35]  # 过滤掉涨跌幅超过35%的异常数据
                time.sleep(0.8) # 基础积分建议增加睡眠时间
                return df
        except Exception as e:
            if "最多访问" in str(e): # 如果是频率限制，多睡一会儿.睡满一分钟后再试（4*15秒）
                time.sleep(LIMIT_SLEEP_SECONDS)
            else:
                print(f"日期 {date} 第 {i+1} 次尝试失败: {e}") #网络问题，再试一次
                time.sleep(2)
    return None
def fetch_basic_by_single_date(date):  ## 增加重试机制，避免偶尔的网络问题导致数据缺失
    for i in range(4):  # 最多重试4次
        try:
            pro = _get_pro_client()
            df = pro.daily_basic(trade_date=date)
            if df is not None:
                time.sleep(1.5) # 基础积分建议增加睡眠时间
                return df
        except Exception as e:
            if "最多访问" in str(e): # 如果是频率限制，多睡一会儿.睡满一分钟后再试（4*15秒）
                time.sleep(LIMIT_SLEEP_SECONDS)
            else:
                print(f"日期 {date} 第 {i+1} 次尝试失败: {e}") #网络问题，再试一次
                time.sleep(2)
    return None
def get_data_by_date(single_function, table_name):
    """
    按交易日多线程循环获取股票日频数据，自动保存数据，返回为空
    
    Args:
        single_function: 下载单日数据的函数
        table_name: DuckDB表名
    """    
    # 过滤出需要下载的日期
    dates_to_download = get_dates_todo(table_name)
    
    if not dates_to_download:
        print("数据已是最新")
        return 

    print(f"正在下载从{(dates_to_download)[0]}到{(dates_to_download)[-1]}的数据")
    
    results = Parallel(n_jobs=DATA_FETCH_JOBS)(
        delayed(single_function)(date) for date in tqdm(dates_to_download, desc="下载进度"))
    
    # 筛选出非空的数据
    all_df_list = [df for df in results if df is not None and not df.empty]
    
    if all_df_list:
        new_data = pd.concat(all_df_list, ignore_index=True)
        new_data.sort_values(by=['trade_date', 'ts_code'], ascending=False, inplace=True)
        db_utils.write_to_db(new_data, table_name, save_mode='append')
def get_stock_data_by_date():
    """获取股票日频数据"""
    get_data_by_date(fetch_bar_by_single_date, table_name='stock_bar')

def get_daily_basic():
    """获取每日指标，如换手率、量比、PB、PE、PS、股息率、总股本、总市值"""
    get_data_by_date(fetch_basic_by_single_date, table_name='daily_basic')

def get_index_data(index_code):
    """获取指数日频数据"""
    # 首先尝试从本地 DuckDB 中读取，避免每次运行都向 Tushare 发起网络请求并重复写入
    #获取下载日期范围
    dates_to_download = get_dates_todo('index_data', ts_code=index_code)
    if not dates_to_download:
        return db_utils.read_sql(f"SELECT * FROM index_data WHERE ts_code='{index_code}'")
    
    else:
        pro = _get_pro_client()
        df = pro.index_daily(ts_code=index_code, start_date=dates_to_download[0], end_date=dates_to_download[-1])
        if df is not None and not df.empty:
            db_utils.write_to_db(df, 'index_data', save_mode='append')
        index_df = db_utils.read_sql(f"SELECT * FROM index_data WHERE ts_code='{index_code}'")
        return index_df

def get_dates_todo(table_name,ts_code=None,start_date= START_DATE):
    """
    获取需要下载更新数据的时间范围
    str: max_trade_day 为数据库中已有数据的最大交易日
    str: latest_day 为当前最新的交易日（根据当前时间自动计算）
    str: table_max_day 为数据库表中已有数据的最大日期记录
    trade_days 为交易日历
    Args:
        table_name: 数据库表名
        ts_code: 股票代码（可选）

    Returns:
        dates_to_download: 需要更新的日期列表，若无须更新则返回None
    """
    now = datetime.now()
    
    # 匹配tushare数据入库时间：大于等于18点算当日，否则停在昨天
    if now.hour >= 18:
        latest_date = now.date()
    else:
        latest_date = now.date() - timedelta(days=1)
        
    # 若为周末，调整至周五
    if latest_date.weekday() >= 5:
        latest_date -= timedelta(days=latest_date.weekday() - 4) #周六的weekday()是5，周日是6，调整到周五就是减去1或2天
        
    latest_day = latest_date.strftime('%Y%m%d')

    trade_day_path = Path(f'{BASIC_INFO_PATH}/trade_day.csv')
    trade_day_path.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
    # 初始化读取交易日历并比较
    if trade_day_path.exists():
        trade_days = pd.read_csv(trade_day_path)
        max_trade_day = trade_days['cal_date'].astype(str).max()
    else:
        max_trade_day = '0'
        
    # 当前最新日期latest_day 超出日历已有最大日期max_trade_day，调用get_trade更新max_trade_day
    if latest_day > max_trade_day:
        trade_days = get_trade(start_date, latest_day)
        max_trade_day = trade_days['cal_date'].astype(str).max()
        
    # 获取数据表中最新的日期记录
    try:
        if ts_code is not None:
            query = f"SELECT MAX(trade_date) as max_date FROM {table_name} WHERE ts_code='{ts_code}'"
        else:
            query = f"SELECT MAX(trade_date) as max_date FROM {table_name}"
        result = db_utils.read_sql(query)
        table_max_day = str(result.iloc[0, 0])
    except RuntimeError:
        # db_utils 已将锁定/占用等严重错误包装为 RuntimeError，直接向上抛出
        raise
    except Exception:
        # 表不存在是首次运行的正常情况，视为无历史数据
        table_max_day = '0'
    
    # 若表数据落后于最新交易日，则需要返回需要更新的日期列表，下载最新交易数据
    if table_max_day < max_trade_day:
        dates_to_download_df =trade_days[trade_days['cal_date'].astype(str) > table_max_day]
        dates_to_download = dates_to_download_df['cal_date'].astype(str).tolist()
        return sorted(dates_to_download)

def get_basic():
    """获取股票基本信息，本地有数据且未超过一年则直接返回，否则从 tushare 重新拉取"""
    last_fetch = get_last_fetch_date('stock_basic')
    if last_fetch:
        last_fetch_dt = datetime.strptime(last_fetch, '%Y%m%d')
        if datetime.now() - last_fetch_dt < timedelta(days=BASIC_RENEW_DAYS):
            return db_utils.read_sql('SELECT * FROM stock_basic')
    pro = _get_pro_client()
    df = pro.stock_basic()
    db_utils.write_to_db(df, 'stock_basic', save_mode='replace')
    set_last_fetch_date('stock_basic')
    return df

def fetch_finan_by_single_stock(ts_code):
    for i in range(4):  # 最多重试4次
        try:
            pro = _get_pro_client()
            df = pro.fina_indicator(ts_code=ts_code)
            if df is not None:
                time.sleep(0.8)  # 财务接口频率限制较严，适当放慢
                return df
        except Exception as e:
            if '最多访问' in str(e):
                time.sleep(LIMIT_SLEEP_SECONDS)
                # 四次尝试,每个核心睡满一分钟
            else:
                print(f"{ts_code} 第 {i+1} 次尝试失败: {e}")
                time.sleep(1)
    return None
def get_last_fetch_date(table_name):
    """从 fetch_log.csv 读取指定表的上次抓取日期，不存在返回 None"""
    if not os.path.exists(FETCH_LOG_PATH):
        return None
    df = pd.read_csv(FETCH_LOG_PATH, dtype=str)
    row = df[df['table_name'] == table_name]
    if row.empty:
        return None
    return row.iloc[0]['last_fetch_date']

def set_last_fetch_date(table_name):
    """将指定表的抓取日期更新为今天，写入 fetch_log.csv"""
    today = datetime.now().strftime('%Y%m%d')
    if os.path.exists(FETCH_LOG_PATH):
        df = pd.read_csv(FETCH_LOG_PATH, dtype=str)
        df = df[df['table_name'] != table_name] # 删除已有的行,保留别的表的更新时间
    else:
        df = pd.DataFrame(columns=['table_name', 'last_fetch_date'])
    new_row = pd.DataFrame({'table_name': [table_name], 'last_fetch_date': [today]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FETCH_LOG_PATH, index=False)

def get_financial():
    """获取全量财务数据，每6个月更新一次。从 stock_basic 读取股票列表，并行按股票抓取后统一写库。
    因为tushare的限制,导致下载的时间很久,大约30分钟
    """
    last_fetch = get_last_fetch_date('fina_indicator')
    if last_fetch:
        last_fetch_dt = datetime.strptime(last_fetch, '%Y%m%d')
        if datetime.now() - last_fetch_dt < timedelta(days=FINANCIAL_RENEW_DAYS):
            print(f"财务数据无需更新，上次更新于 {last_fetch}")
            return

    stocks_df = get_basic()
    ts_codes = stocks_df['ts_code'].tolist()
    print(f"开始下载 {len(ts_codes)} 只股票的财务数据...")

    results = Parallel(n_jobs=DATA_FETCH_JOBS)(
        delayed(fetch_finan_by_single_stock)(code) for code in tqdm(ts_codes, desc="下载进度"))

    all_df = [df for df in results if df is not None and not df.empty]
    if all_df:
        new_data = pd.concat(all_df, ignore_index=True)
        db_utils.write_to_db(new_data, 'fina_indicator', save_mode='replace')
        set_last_fetch_date('fina_indicator')
def get_industry():
    last_fetch = get_last_fetch_date('sw_industry')
    if last_fetch:
        last_fetch_dt = datetime.strptime(last_fetch, '%Y%m%d')
        if datetime.now() - last_fetch_dt < timedelta(days=INDUSTRY_RENEW_DAYS):
            return db_utils.read_sql('SELECT * FROM sw_industry')
    # 本地没有或已经超过指定天数，则从 tushare 获取
    all_data = []
    limit = 3000  # 设置单次获取上限
    offset = 0    # 初始偏移量为0
    pro = _get_pro_client()
    while True:
        # 传入分页参数
        df = pro.index_member_all(limit=limit, offset=offset)
        # 如果返回的 DataFrame 为空，说明已经取到了最后
        if df.empty:
            print("已获取所有数据。")
            break
            
        all_data.append(df)
        print(f"已获取偏移量 {offset} 开始的 {len(df)} 条数据")
        # 增加偏移量，准备取下一页
        offset += limit
        
    # 合并所有数据
    df = pd.concat(all_data, ignore_index=True)
    db_utils.write_to_db(df, 'sw_industry', save_mode='replace')
    set_last_fetch_date('sw_industry')
    return df

# if __name__ == "__main__":
    # pro = _get_pro_client()
    # df = pro.index_member_all()
    # df.to_csv("test.csv")

