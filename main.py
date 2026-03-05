# -*- coding: utf-8 -*-
"""
主程序 - 数据获取和流程控制
"""
import tushare as ts
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timedelta
from tqdm import tqdm
from factor_construct import compute_week_effect, compute_pricing_factors
from factor_analyze import evaluate_factor
from joblib import Parallel, delayed
import os
import db_utils

# ====================  路径常量配置  ====================
DATA_ROOT = './Data'
MARKET_PATH = f'{DATA_ROOT}/Stock_data/Market'
FINANCIALS_PATH = f'{DATA_ROOT}/Stock_data/Financials'
DERIVED_FACTORS_PATH = f'{DATA_ROOT}/Processed/Derived_Factors'
STOCK_UNIVERSE_PATH = f'{DATA_ROOT}/Metadata'
TRADING_CALENDAR_PATH = f'{DATA_ROOT}/Metadata'
INDEX_DATA_PATH = f'{DATA_ROOT}/Metadata/Market'
OUTPUT_PATH = './Output'

# Tushare配置
TUSHARE_TOKEN = 'ed4aeb5f1c3f85cace382275ee12aa23cc4ece2e9d1195dbfa55bba4'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


# ====================  数据获取函数  ====================
def get_ins(index_code):
    """获取指数成分股并保存"""
    today = datetime.now()
    if today.month == 1:
        first_day_of_last_month = today.replace(year=today.year - 1, month=12, day=1)
    else:
        first_day_of_last_month = today.replace(month=today.month - 1, day=1)
    last_day_of_last_month = today.replace(day=1) - timedelta(days=1)
    
    start = first_day_of_last_month.strftime('%Y%m%d')
    end = last_day_of_last_month.strftime('%Y%m%d')
    
    try:
        ins = pro.index_weight(index_code=index_code, start_date=start, end_date=end)
        if ins is None or ins.empty:
            print(f"警告: 未获取到指数 {index_code} 的成分股数据")
            return
        ins_codes = ins['con_code'].unique().tolist()
        print(f"指数 {index_code} 的唯一股票代码数量：{len(ins_codes)}")
        pd.DataFrame({'con_code': ins_codes}).to_csv(f'{STOCK_UNIVERSE_PATH}/{index_code}_ins.csv', index=False)
        print(f"成分股代码已保存到 {STOCK_UNIVERSE_PATH}/{index_code}_ins.csv")
    except Exception as e:
        print(f"获取指数成分股失败: {e}")


def get_trade(start_date, end_date):
    """获取交易日历"""
    df = pro.trade_cal(exchange='SSE', is_open='1', start_date=start_date, 
                      end_date=end_date, fields='cal_date')
    df.to_csv(f'{TRADING_CALENDAR_PATH}/trade_day.csv', index=False)
    print(f"交易日历已保存到 {TRADING_CALENDAR_PATH}/trade_day.csv")
def fetch_bar_by_single_date(date, token):
    """获取单个交易日的股票日线数据"""
    try:
        ts.set_token(token)
        pro_local = ts.pro_api()
        df = pro_local.daily(trade_date=date)
        time.sleep(0.15)  # 避免频率限制
        return (date, df)
    except Exception as e:
        return (date, None)

def fetch_basic_by_single_date(date, token):
    """获取单个交易日的股票每日指标"""
    try:
        ts.set_token(token)
        pro_local = ts.pro_api()
        df = pro_local.daily_basic(trade_date=date)
        time.sleep(0.15)  # 避免频率限制
        return (date, df)
    except Exception as e:
        return (date, None)
## 写一个通用的按交易日，线程下载数据的函数
def get_data_by_date(single_function, output, table_name=None, use_db=True):
    """
    获取股票日频数据（多线程）
    
    Args:
        single_function: 下载单日数据的函数
        output: CSV输出路径（兼容模式）
        table_name: DuckDB表名
        use_db: 是否使用DuckDB存储
    """
    trade_dates = pd.read_csv(f'{TRADING_CALENDAR_PATH}/trade_day.csv')
    all_dates = trade_dates['cal_date'].astype(str).tolist()
    
    # 检查是否有已下载的数据
    already_downloaded = set()
    
    if use_db and table_name:
        # 从数据库检查已有数据
        try:
            query = f"SELECT DISTINCT trade_date FROM {table_name}"
            existing_data = db_utils.read_sql(query)
            already_downloaded = set(existing_data['trade_date'].astype(str).unique())
            print(f"发现已有数据，包含 {len(already_downloaded)} 个交易日")
        except:
            print(f"数据库表 {table_name} 为空或不存在，将创建新表")
    elif os.path.exists(output):
        # 从CSV检查已有数据
        try:
            existing_data = pd.read_csv(output)
            already_downloaded = set(existing_data['trade_date'].astype(str).unique())
            print(f"发现已有数据，包含 {len(already_downloaded)} 个交易日")
        except:
            pass
    
    # 过滤出需要下载的日期
    dates_to_download = [d for d in all_dates if d not in already_downloaded]
    
    if not dates_to_download:
        print("所有数据已下载完成！")
        return
    
    print(f"准备下载 {len(dates_to_download)} 个交易日的股票数据...")
    
    # 使用threading避免pickle问题
    results = Parallel(n_jobs=7, prefer="threads")(
        delayed(single_function)(date, TUSHARE_TOKEN) for date in tqdm(dates_to_download, desc="下载进度")
    )
    
    # 过滤掉None结果
    all_df = [df for date, df in results if df is not None and not df.empty]
    
    if all_df:
        new_data = pd.concat(all_df, ignore_index=True)
        new_data.sort_values(by=['trade_date', 'ts_code'], ascending=False, inplace=True)
        
        if use_db and table_name:
            # 保存到数据库
            print(f"\n新增 {len(new_data)} 条记录，正在写入数据库...")
            db_utils.write_to_db(new_data, table_name, if_exists='append')
            print(f"已保存到 DuckDB 表: {table_name}")
        else:
            # 保存到CSV
            if os.path.exists(output) and len(already_downloaded) > 0:
                try:
                    existing_data = pd.read_csv(output)
                    final_data = pd.concat([existing_data, new_data], ignore_index=True)
                    print(f"\n新增 {len(new_data)} 条记录，合并后共 {len(final_data)} 条记录")
                except:
                    final_data = new_data
                    print(f"\n获取 {len(new_data)} 条新记录")
            else:
                final_data = new_data
                print(f"\n成功获取 {len(new_data)} 条记录")
            final_data.to_csv(output, index=False)
            print(f"已保存到 {output}")
    else:
        print("本次未获取到新数据")
def get_stock_data_by_date(output, use_db=True):
    """获取股票日频数据"""
    get_data_by_date(fetch_bar_by_single_date, output, table_name='stock_bar', use_db=use_db)

def get_index_data(index_code):
    """获取指数日频数据"""
    df = pro.index_daily(ts_code=index_code)
    if use_db:
        db_utils.write_to_db(df, 'index_data', if_exists='append')
        print(f"指数日行情已保存到 DuckDB")
    else:
        df.to_csv(f'{INDEX_DATA_PATH}/{index_code}.csv', index=False)
        print(f"指数日行情已保存到 {INDEX_DATA_PATH}/{index_code}.csv")

def get_daily_basic(output, use_db=True):
    """获取每日指标，如换手率、量比、PB、PE、PS、股息率、总股本、总市值"""
    get_data_by_date(fetch_basic_by_single_date, output, table_name='daily_basic', use_db=use_db)
    
# ====================  主流程  ====================
def main():
    """主流程 - 一键运行"""
    global use_db
    use_db = True  # 是否使用DuckDB (推荐设为True)
    
    index_code = '000852.SH'    #'399852.SZ'    # 中证1000    #'000016.SH'  # 上证50
    start_date = '20160101'
    end_date = datetime.now().strftime("%Y%m%d")
    ## 对于个股数据，按日增量获取。指数、交易日数据全量获取

    # 1. 获取成分股
    print("\n获取指数成分股...")
    get_ins(index_code)
    
    # 2. 获取交易日
    print("\n获取交易日历...")
    get_trade(start_date, end_date)
    
    # 3. 获取全量股票日线（Tushare更推荐按日循环，所以此处不对股票进行筛选）
    # 不用指定日期，会在代码内部自动填补最新数据
    print("\n获取股票日线数据...")
    get_stock_data_by_date(f'{MARKET_PATH}/stock_bar.csv', use_db=use_db)
    
    # 4. 获取指数日线
    print("\n获取指数日线数据...")
    get_index_data(index_code)
    
    ## 5. 获取补充数据并计算因子
    print("\n获取补充数据（如PB、MV）...")
    get_daily_basic(f'{FINANCIALS_PATH}/daily_basic.csv', use_db=use_db)
    
    # 6. 定价因子计算
    print("\n计算定价因子（SMB、HML、UMD）...")
    compute_pricing_factors(
        stock_path=f'{MARKET_PATH}/stock_bar.csv',
        financial_path=f'{FINANCIALS_PATH}/daily_basic.csv',
        out_dir=DERIVED_FACTORS_PATH,
        use_db=use_db
    )
    
    # 7. 计算周末效应因子
    print("\n计算周末效应因子...")
    compute_week_effect()
    
    # # 8. 因子评测（月度调仓，使用DuckDB）
    print("\n因子评测...")
    res = evaluate_factor(
        factor_table='week_factor',      # DuckDB表名
        index_code='000852.SH',          # 中证1000
        output_path=OUTPUT_PATH,
        n_groups=10
    )

if __name__ == '__main__':
    main()
    
















