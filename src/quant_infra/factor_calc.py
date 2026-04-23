import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from quant_infra import db_utils, get_data
from quant_infra.const import RESID_REG_WINDOW, SPEC_VOL_WINDOW
from datetime import timedelta

#按日期计算定价因子   
# 定义单日计算函数
def calc_single_pricing_factors(trade_date, day_df):
    """计算单个交易日的定价因子
    return:dict 包含 trade_date, MKT, SMB, HML, UMD
    """
    # 过滤有效数据
    valid_data = day_df.dropna(subset=['pct_chg'])
    
    if len(valid_data) < 9:  # 至少需要9只股票才能分3组
        return None
        
    ## MKT因子：当日所有股票的平均收益率
    mkt_factor = valid_data['pct_chg'].mean()
    
    factors_dict = {'SMB': np.nan, 'HML': np.nan, 'UMD': np.nan}
    # 配置格式: (因子名称, 排序基于的列名, 是否反转计算_即用低组减去高组)
    factor_configs = [
        ('SMB', 'month_mv', True),   # 小减大 (前33% - 后33%) 
        ('HML', 'month_pb', True),   # 价值减成长 (低PB即前33% - 高PB即后33%)
        ('UMD', 'month_ret', False)  # 赢家减输家 (高收益即后33% - 低收益即前33%)
    ]
    
    for factor_name, col, low_minus_high in factor_configs:
        sub_data = valid_data.dropna(subset=[col])
        if len(sub_data) >= 3:
            sub_data = sub_data.sort_values(col)
            n_third = len(sub_data) // 3
            ret_low = sub_data.iloc[:n_third]['pct_chg'].mean()
            ret_high = sub_data.iloc[-n_third:]['pct_chg'].mean()
            # low_minus_high 为 True 时：用 低组（前33%） 减 高组（后33%）
            factors_dict[factor_name] = (ret_low - ret_high) if low_minus_high else (ret_high - ret_low)
    
    return {
        'trade_date': int(trade_date),
        'MKT': mkt_factor,
        **factors_dict  ## 字典解包
    }
def compute_pricing_factors():
    """
    计算定价因子（MKT、SMB、HML、UMD）
    依据 README 中定价因子计算部分的步骤：
    1. SMB: 上一月底的流动市值排名后三分之一股票组合的收益减去前三分之一股票组合的收益
    2. HML: 上一月底按账面市值比前三分之一股票组合的收益减去后三分之一股票组合的收益
    3. UMD: 上一月底按当月累积收益排名前三分之一股票组合的收益减去后三分之一股票组合的收益
    4. MKT: 当日所有股票的平均收益率
    """    
    dates_to_download = get_data.get_dates_todo('pricing_factors')

    if not dates_to_download:
        print("定价因子数据已是最新")
        return 
    
    print('开始计算新的定价因子')
    # 过滤出需要计算的日期，由于是几个定价因子都是依据上一个月末的数据算的，所以要有“上个月第一个天————最新日期的完整数据”
    # 获得上一个月第一天的方式与get_ins中的一样
    last_day_of_last_month = pd.to_datetime(str(dates_to_download[0]), format='%Y%m%d').replace(day=1) - timedelta(days=1)
    first_month_date_str = last_day_of_last_month.replace(day=1).strftime('%Y%m%d')
    last_date = dates_to_download[-1]

    # 使用 SQL 直接合并股票数据和财务数据
    query = f"""
    SELECT b.ts_code, b.trade_date, b.pct_chg, d.total_mv, d.pb
    FROM stock_bar b
    INNER JOIN daily_basic d 
    ON b.ts_code = d.ts_code AND b.trade_date = d.trade_date
    WHERE b.trade_date >= '{first_month_date_str}' AND b.trade_date <= '{last_date}'
    """
    df = db_utils.read_sql(query)

    if len(df) == 0:
        print('daily_basic和 stock_bar 合并后没有数据，检查两表的最新数据是否下载成功')
        return

    # 3. 添加年月标识（用于分组，基于trade_date）
    df['date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    df['year_month'] = df['date'].dt.to_period('M')
    
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    # 1. 先计算出每只股票每个月唯一的指标（月度表）
    monthly_df = df.groupby(['ts_code', 'year_month']).agg({
        'pct_chg': 'sum',
        'total_mv': 'last',
        'pb': 'last'
    }).reset_index()

    # 2. monthly_df 已经是原有的df按月聚合的表了，也就是所有股票，每月一行。
    # 现在对它进行按股票分组，为每个股票的全部月数据，进行 shift(1) 才是真正的上个月，得到上月末的指标值
    # 这样计算定价因子时才是真正的用的上月末数据，不会出现未来数据问题
    # 必须先按股票和时间排序，再按股票分组 shift
    monthly_df = monthly_df.sort_values(['ts_code', 'year_month'])
    monthly_df['month_ret'] = monthly_df.groupby('ts_code')['pct_chg'].shift(1)
    monthly_df['month_mv'] = monthly_df.groupby('ts_code')['total_mv'].shift(1)
    monthly_df['month_pb'] = monthly_df.groupby('ts_code')['pb'].shift(1)
    ## 注意：这里的 month_ret、month_mv、month_pb 都是上月末的值了，后续计算定价因子时就不会有未来数据问题了
    ## 也可以用monthly_df['target_month'] = monthly_df['year_month'] + 1，后续用target_month来merge，但直接shift(1)更简单直接

    # 3. 将这些“上月值”合并回原有的日线 df
    # 删掉月度表里原本的当月值列（由日数据聚合得到的），避免重名冲突
    monthly_df = monthly_df[['ts_code', 'year_month', 'month_ret', 'month_mv', 'month_pb']]
    df = pd.merge(df[['ts_code', 'trade_date', 'pct_chg', 'year_month']], monthly_df, on=['ts_code', 'year_month'], how='left')
    df.dropna(subset=['month_ret', 'month_mv', 'month_pb'], inplace=True)
    ## 只保留需要更新的列，防止后续在月内写入重复的列
    df = df[df['trade_date'].isin(dates_to_download)]
    # 5.按交易日分组，并行计算每个交易日的定价因子
    daily_groups = list(df.groupby('trade_date'))
    
    results = Parallel(n_jobs=-1)(
        delayed(calc_single_pricing_factors)(trade_date, day_df) 
        for trade_date, day_df in tqdm(daily_groups, desc='计算定价因子')
    )
    
    # 过滤掉None结果
    pricing_factors = [dic for dic in results if dic is not None]
        
    # 输出结果
    result_df = pd.DataFrame(pricing_factors)
    result_df = result_df.dropna(subset=['MKT', 'SMB', 'HML', 'UMD'])  # 过滤掉因子值为NaN的行
    # 将结果写入数据库
    db_utils.write_to_db(result_df, 'pricing_factors', save_mode='append')
    return 

def calc_single_resid_rolling(code, stock_df, reg_window=RESID_REG_WINDOW):
    """按股票滚动回归，计算每日残差。

    对于每个交易日 t，使用最近 reg_window 个交易日（含 t）的数据回归四因子模型，
    仅保留 t 当天对应的残差。这样 beta 会随时间滚动更新，但无需单独落库。
    """
    try:
        stock_df = stock_df.sort_values('trade_date').reset_index(drop=True)
        if len(stock_df) < reg_window:
            return pd.DataFrame()

        X = stock_df[['MKT', 'SMB', 'HML', 'UMD']].to_numpy(dtype=float)
        y = stock_df['pct_chg'].to_numpy(dtype=float)
        resid = np.full(len(stock_df), np.nan, dtype=float)

        for end_idx in range(reg_window - 1, len(stock_df)):
            start_idx = end_idx - reg_window + 1
            X_win = X[start_idx:end_idx + 1]
            y_win = y[start_idx:end_idx + 1]
            X_reg = np.column_stack([np.ones(len(X_win)), X_win])
            beta_vec = np.linalg.lstsq(X_reg, y_win, rcond=None)[0]
            resid[end_idx] = y_win[-1] - (X_reg[-1] @ beta_vec)

        result = stock_df[['ts_code', 'trade_date']].copy()
        result['resid'] = resid
        return result.dropna(subset=['resid'])
    except Exception as e:
        # 返回空结果避免中断整体流程，同时保留错误上下文
        print(f"calc_single_resid_rolling 失败: ts_code={code}, error={type(e).__name__}: {e}")
        return pd.DataFrame()

def calc_resid():
    """
    使用滚动回归计算日度残差，并写入 stock_resids。

    对于每个交易日 t：
    1. 取最近 RESID_REG_WINDOW 个交易日（含 t）的股票收益与四因子收益；
    2. 现场回归得到当期 beta；
    3. 仅保留 t 当天残差。
    """
    compute_pricing_factors()
    dates_to_download = get_data.get_dates_todo('stock_resids')
    if not dates_to_download:
        print("残差数据已是最新")
        return

    reg_window = RESID_REG_WINDOW
    # 取足够长的自然日缓冲，覆盖滚动回归窗口。
    reg_buffer_days = max(60, int(np.ceil(reg_window * 1.6)))
    start_dt = pd.to_datetime(str(dates_to_download[0]), format='%Y%m%d') - timedelta(days=reg_buffer_days)
    start_str = start_dt.strftime('%Y%m%d')

    # 把定价因子和股票日线数据合并在一起，减少后续计算时的重复读取和合并
    query = f"""
    SELECT b.ts_code, b.trade_date, b.pct_chg, p.MKT, p.SMB, p.HML, p.UMD
    FROM stock_bar b
    LEFT JOIN (SELECT trade_date, MKT, SMB, HML, UMD FROM pricing_factors) p
    ON b.trade_date = p.trade_date
    WHERE b.trade_date >= '{start_str}' AND b.trade_date <= '{dates_to_download[-1]}'
    ORDER BY b.ts_code, b.trade_date
    """
    df = db_utils.read_sql(query)

    # 过滤掉收益率或定价因子缺失的行
    df = df.dropna(subset=['pct_chg', 'MKT', 'SMB', 'HML', 'UMD'])

    groups = df.groupby('ts_code')

    resid_results = Parallel(n_jobs=-1)(
        delayed(calc_single_resid_rolling)(code, group_df, reg_window)
        for code, group_df in tqdm(groups, desc='滚动计算残差')
    )
    resid_results = [x for x in resid_results if x is not None and not x.empty]
    if not resid_results:
        print(f'未生成可用残差：有效样本可能不足 {reg_window} 个交易日')
        return

    all_resid = pd.concat(resid_results, ignore_index=True)
    all_resid['trade_date'] = all_resid['trade_date'].astype(str)
    result = all_resid[all_resid['trade_date'].isin(dates_to_download)].copy()

    if result.empty:
        print(f"没有可保存的残差数据（回归历史可能不足 {reg_window} 个交易日）")
        return

    db_utils.write_to_db(result[['ts_code', 'trade_date', 'resid']], 'stock_resids', save_mode='append')
    print(f"滚动残差计算完成，共 {len(result)} 条记录")

def calc_spec_vol():
    """
    基于 stock_resids 计算特质波动率因子（日频）
    特质波动率 = 近 SPEC_VOL_WINDOW 个交易日残差的波动率 = std(residuals)
    结果存入 spec_vol 表，列为 (ts_code, trade_date, factor)
    """
    dates_to_download = get_data.get_dates_todo('spec_vol')
    if not dates_to_download:
        print("特质波动率因子数据已是最新")
        return

    vol_window = SPEC_VOL_WINDOW
    # 往前多取足够长的自然日作为缓冲，确保能填满残差滚动窗口
    vol_buffer_days = max(30, int(np.ceil(vol_window * 3)))
    start_dt = pd.to_datetime(str(dates_to_download[0]), format='%Y%m%d') - timedelta(days=vol_buffer_days)
    start_str = start_dt.strftime('%Y%m%d')

    query = f"""
    SELECT ts_code, trade_date, resid
    FROM stock_resids
    WHERE trade_date >= '{start_str}' AND trade_date <= '{dates_to_download[-1]}'
    ORDER BY ts_code, trade_date
    """
    df = db_utils.read_sql(query)

    if df.empty:
        print("stock_resids 为空，请先运行 calc_resid()")
        return

    df['trade_date'] = df['trade_date'].astype(str)
    df = df.sort_values(['ts_code', 'trade_date'])

    # 按股票分组，计算滚动波动率
    df['factor'] = df.groupby('ts_code')['resid'].transform(
        lambda x: x.rolling(window=vol_window, min_periods=vol_window).std()
    )

    result = df[df['trade_date'].isin(dates_to_download)][['ts_code', 'trade_date', 'factor']]
    result = result.dropna(subset=['factor'])

    if result.empty:
        print(f"没有可保存的特质波动率数据（残差历史可能不足 {vol_window} 个交易日）")
        return

    db_utils.write_to_db(result, 'spec_vol', save_mode='append')
    print(f"特质波动率因子计算完成，共 {len(result)} 条记录")
def winsorize(series, n=3):
    """按 n 倍标准差缩尾，将超过范围的值替换为边界值"""
    mean, std = series.mean(), series.std()
    return series.clip(mean - n * std, mean + n * std)
