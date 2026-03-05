# -*- coding: utf-8 -*-
"""
因子构建模块 - 所有因子计算逻辑
支持使用DuckDB数据库而不是直接读取CSV文件
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import db_utils

def compute_week_effect():
    """
    计算周末效应原始因子（Carhart四因子模型）- 按README算法优化版
    
    算法步骤（参考README）：
    1. 读取已有的定价因子（MKT, SMB, HML, UMD）并合并为df_pricing_merge
    2. 降维：将日线数据转换为周一、周内其他时间
    4. 使用Joblib多进程并行计算每只股票的残差
    5. 因子 = 周一残差 - 周内残差
    6. 月度聚合
    
    使用DuckDB读取和写入数据
    """   
    # 合并
    idx = db_utils.query_index_data(columns=['trade_date', 'pct_chg'])
    idx["date"] = pd.to_datetime(idx["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    idx = idx[["date", "pct_chg"]].rename(columns={"pct_chg": "MKT"})
    
    pricing_factors = db_utils.read_sql("SELECT trade_date, SMB, HML, UMD FROM pricing_factors")
    pricing_factors['date'] = pd.to_datetime(pricing_factors['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    pricing_factors = pricing_factors[['date', 'SMB', 'HML', 'UMD']].copy()
    
    df_pricing_merge = idx.merge(pricing_factors, on='date', how='inner')
    df_pricing_merge = df_pricing_merge[['date', 'MKT', 'SMB', 'HML', 'UMD']].copy()
    print(f'定价因子数据合并成功: {len(df_pricing_merge)} 个交易日')
    
    # ========== 检查week_factor表中已有哪些股票 ==========
    # print('检查已有的周末效应因子...')
    existing_week_factors = None
    existing_stocks = set()
    try:
        existing_week_factors = db_utils.read_sql("SELECT DISTINCT ts_code FROM week_factor")
        existing_stocks = set(existing_week_factors['ts_code'].tolist())
        # print(f'  已有 {len(existing_stocks)} 只股票的周末效应因子')
    except:
        print('  未找到已有因子数据，将全部计算')
        existing_stocks = set()
    
    # ========== 检查是否已有beta系数 ==========
    # print('检查已有beta系数...')
    existing_betas = None
    try:
        existing_betas = db_utils.read_sql("SELECT * FROM stock_betas WHERE status IS NULL OR status != 'failed'")
        # print(f'  已有 {len(existing_betas)} 只股票的有效beta系数')
        existing_beta_dict = existing_betas.set_index('ts_code').to_dict('index')
    except:
        print('  未找到已有beta数据')
        existing_beta_dict = {}
    
    # 一次性读取所有股票数据（避免并发访问数据库）
    # print('读取所有股票数据...')
    all_stock_data = db_utils.query_stock_bar(columns=['ts_code', 'trade_date', 'pct_chg'])
    all_stock_data["date"] = pd.to_datetime(all_stock_data["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    
    # 提前合并定价因子数据
    all_stock_data = all_stock_data.merge(df_pricing_merge, on='date', how='inner')
    
    # **过滤掉定价因子为NaN的行**
    all_stock_data = all_stock_data.dropna(subset=['MKT', 'SMB', 'HML', 'UMD'])
    
    # print(f'  读取完成，共 {len(all_stock_data)} 条记录（已过滤NaN）')
    
    # 按股票代码分组，字典的查询效率远高于全量DataFrame
    # print('按股票分组数据...')
    stock_groups = dict(list(all_stock_data.groupby('ts_code')))
    print(f'  共 {len(stock_groups)} 只股票有数据')
    
    # print('多线程计算每只股票的因子...')
    def calc_stock_factor_fast(ts_code, stock_df, existing_beta=None):
        """计算单只股票的周末效应因子 - 用全历史数据估计beta，然后按月计算因子"""
        try:
            if len(stock_df) == 0:
                return {'factors': [], 'beta': None, 'status': '交易日数不足'}
            if len(stock_df) < 50:  # 至少需要50个交易日
                return {'factors': [], 'beta': None, 'status': '交易日数不足'}

            # 添加时间标识
            stock_df = stock_df.copy()  # 避免修改原数据
            stock_df['weekday'] = stock_df['date'].dt.weekday
            stock_df['week'] = stock_df['date'].dt.to_period('W-FRI')
            stock_df['year_month'] = stock_df['date'].dt.to_period('M')
            stock_df['is_monday'] = (stock_df['weekday'] == 0).astype(int)

            # 分离周一和周内数据
            monday_df = stock_df[stock_df['is_monday'] == 1].sort_values(by='date')
            weekday_df = stock_df[stock_df['is_monday'] == 0].sort_values(by='date')
            
            if len(monday_df) < 10 or len(weekday_df) < 20:  # 提高数据量要求
                return {'factors': [], 'beta': None, 'status': '交易日数不足'}
            
            # 检查是否使用已有beta
            if existing_beta is not None and not pd.isna(existing_beta.get('monday_intercept')):
                # 使用已有的beta系数
                beta_monday = np.array([existing_beta['monday_intercept'], existing_beta['monday_mkt'], 
                                       existing_beta['monday_smb'], existing_beta['monday_hml'], existing_beta['monday_umd']])
                beta_weekday = np.array([existing_beta['weekday_intercept'], existing_beta['weekday_mkt'],
                                        existing_beta['weekday_smb'], existing_beta['weekday_hml'], existing_beta['weekday_umd']])
                
                # 检查beta是否有效
                if not (np.isfinite(beta_monday).all() and np.isfinite(beta_weekday).all()):
                    return {'factors': [], 'beta': None, 'status': '未知失败'}
                    
                beta_info = None  # 不需要重新保存
            else:
                # 构建回归矩阵（手动加截距项）
                X_monday = np.column_stack([np.ones(len(monday_df)), monday_df[['MKT', 'SMB', 'HML', 'UMD']].to_numpy()])
                y_monday = monday_df['pct_chg'].to_numpy()
                X_weekday = np.column_stack([np.ones(len(weekday_df)), weekday_df[['MKT', 'SMB', 'HML', 'UMD']].to_numpy()])
                y_weekday = weekday_df['pct_chg'].to_numpy()
                
                # 检查数据是否有效
                if not (np.isfinite(X_monday).all() and np.isfinite(y_monday).all() and 
                       np.isfinite(X_weekday).all() and np.isfinite(y_weekday).all()):
                    return {'factors': [], 'beta': None, 'status': '未知失败'}

                # 估计beta（用全历史数据），增加错误处理
                try:
                    beta_monday = np.linalg.lstsq(X_monday, y_monday, rcond=None)[0]  # 更稳定的最小二乘法
                    beta_weekday = np.linalg.lstsq(X_weekday, y_weekday, rcond=None)[0]
                except (np.linalg.LinAlgError, ValueError) as e:
                    # print(f"{ts_code}: 回归失败 - {e}")
                    return {'factors': [], 'beta': None, 'status': '未知失败'}
                
                # 再次检查beta是否有效
                if not (np.isfinite(beta_monday).all() and np.isfinite(beta_weekday).all()):
                    return {'factors': [], 'beta': None, 'status': '未知失败'}
                
                # 保存beta信息
                beta_info = {
                    'ts_code': ts_code,
                    'monday_intercept': float(beta_monday[0]),
                    'monday_mkt': float(beta_monday[1]),
                    'monday_smb': float(beta_monday[2]),
                    'monday_hml': float(beta_monday[3]),
                    'monday_umd': float(beta_monday[4]),
                    'weekday_intercept': float(beta_weekday[0]),
                    'weekday_mkt': float(beta_weekday[1]),
                    'weekday_smb': float(beta_weekday[2]),
                    'weekday_hml': float(beta_weekday[3]),
                    'weekday_umd': float(beta_weekday[4]),
                    'update_date': pd.Timestamp.now().strftime('%Y%m%d')
                }
            
            # 构建X矩阵用于计算残差
            X_monday = np.column_stack([np.ones(len(monday_df)), monday_df[['MKT', 'SMB', 'HML', 'UMD']].to_numpy()])
            y_monday = monday_df['pct_chg'].to_numpy()
            X_weekday = np.column_stack([np.ones(len(weekday_df)), weekday_df[['MKT', 'SMB', 'HML', 'UMD']].to_numpy()])
            y_weekday = weekday_df['pct_chg'].to_numpy()
            
            # 计算残差
            monday_df = monday_df.copy()
            weekday_df = weekday_df.copy()
            monday_df['resid'] = y_monday - X_monday @ beta_monday
            weekday_df['resid'] = y_weekday - X_weekday @ beta_weekday
            
            # 按月聚合残差
            results = []
            for year_month in stock_df['year_month'].unique():
                monday_month = monday_df[monday_df['year_month'] == year_month]
                weekday_month = weekday_df[weekday_df['year_month'] == year_month]
                
                if len(monday_month) > 0 and len(weekday_month) > 0:
                    # 周末效应因子 = 周一残差均值 - 周内残差均值
                    raw_factor = monday_month['resid'].mean() - weekday_month['resid'].mean()
                    
                    results.append({
                        'ts_code': ts_code,
                        'year_month': str(year_month),
                        'trade_date': monday_month['trade_date'].max(),  # 使用该月最后一个交易日
                        'raw_factor': raw_factor
                    })
            
            return {'factors': results, 'beta': beta_info, 'status': 'success'}
            
        except Exception as e:
            # 完全静默，不输出任何错误
            return {'factors': [], 'beta': None, 'status': '未知失败'}
    
    # 获取需要计算因子的股票（week_factor中没有的）
    all_codes = list(stock_groups.keys())
    need_calc_codes = [code for code in all_codes if code not in existing_stocks]
    # print(f'  共 {len(all_codes)} 只股票有数据')
    print(f'  其中 {len(existing_stocks)} 只已有因子，{len(need_calc_codes)} 只需要计算')
    
    if len(need_calc_codes) == 0:
        print('所有股票的因子已存在！')
        return db_utils.read_sql("SELECT * FROM week_factor")
    
    # 对于需要计算的股票，检查是否有beta
    need_beta_codes = [code for code in need_calc_codes if code not in existing_beta_dict]
    print(f'  其中 {len(need_beta_codes)} 只股票需要先计算beta')
    
    # 第一步：计算缺失的beta
    if len(need_beta_codes) > 0:
        # print(f'并行计算 {len(need_beta_codes)} 只股票的beta...')
        beta_results = Parallel(n_jobs=-1, prefer='threads', verbose=0)(
            delayed(calc_stock_factor_fast)(code, stock_groups[code], None)
            for code in tqdm(need_beta_codes, desc='计算beta', ncols=80, position=0, leave=True)
        )
        
        # 保存新计算的beta
        new_betas = []
        for i, code in enumerate(need_beta_codes):
            if beta_results[i]['beta'] is not None:
                new_betas.append(beta_results[i]['beta'])
            else:
                # 失败的也标记，使用具体的失败原因
                status = beta_results[i].get('status', '未知失败')
                new_betas.append({
                    'ts_code': code,
                    'monday_intercept': np.nan, 'monday_mkt': np.nan, 'monday_smb': np.nan,
                    'monday_hml': np.nan, 'monday_umd': np.nan,
                    'weekday_intercept': np.nan, 'weekday_mkt': np.nan, 'weekday_smb': np.nan,
                    'weekday_hml': np.nan, 'weekday_umd': np.nan,
                    'update_date': pd.Timestamp.now().strftime('%Y%m%d'),
                    'status': status
                })
        
        if len(new_betas) > 0:
            new_beta_df = pd.DataFrame(new_betas)
            success_count = len([b for b in new_betas if b.get('status') is None or b.get('status') == 'success'])
            insufficient_count = len([b for b in new_betas if b.get('status') == '交易日数不足'])
            unknown_fail_count = len([b for b in new_betas if b.get('status') == '未知失败'])
            print(f'保存beta: 成功 {success_count} 只, 交易日数不足 {insufficient_count} 只, 未知失败 {unknown_fail_count} 只')
            
            # 合并保存
            if existing_betas is not None and len(existing_betas) > 0:
                all_betas_df = pd.concat([existing_betas, new_beta_df], ignore_index=True)
            else:
                all_betas_df = new_beta_df
            db_utils.write_to_db(all_betas_df, 'stock_betas', if_exists='replace')
            
            # 更新existing_beta_dict
            for beta in new_betas:
                if beta.get('status') != 'failed':
                    existing_beta_dict[beta['ts_code']] = beta
    
    # 第二步：计算所有需要的因子
    print(f'计算 {len(need_calc_codes)} 只股票的周末效应因子...')
    factor_results = Parallel(n_jobs=-1, prefer='threads', verbose=0)(
        delayed(calc_stock_factor_fast)(code, stock_groups[code], existing_beta_dict.get(code))
        for code in tqdm(need_calc_codes, desc='计算因子', ncols=80, position=0, leave=True)
    )
    
    # 合并结果，只收集因子
    all_factors = []
    for result in factor_results:
        all_factors.extend(result['factors'])
    
    if len(all_factors) == 0:
        print('警告：没有成功计算出任何因子！')
        return existing_week_factors if existing_week_factors is not None else pd.DataFrame()
    
    factor_df = pd.DataFrame(all_factors)
    
    # ========== 月度聚合 ==========
    print('月度聚合...')
    new_monthly_factor = factor_df.groupby(['ts_code', 'year_month'], as_index=False).agg({'raw_factor': 'mean','trade_date': 'last'})
    
    new_monthly_factor = new_monthly_factor.sort_values(['ts_code', 'year_month']).reset_index(drop=True)
    
    # 合并新旧因子数据
    if existing_week_factors is not None and len(existing_stocks) > 0:
        print(f'合并新旧因子数据...')
        old_factors = db_utils.read_sql("SELECT * FROM week_factor")
        monthly_factor = pd.concat([old_factors, new_monthly_factor], ignore_index=True)
        monthly_factor = monthly_factor.drop_duplicates(subset=['ts_code', 'year_month'], keep='last')
        monthly_factor = monthly_factor.sort_values(['ts_code', 'year_month']).reset_index(drop=True)
    else:
        monthly_factor = new_monthly_factor
    
    # 保存结果
    print('保存到DuckDB...')
    db_utils.write_to_db(monthly_factor, 'week_factor', if_exists='replace')
    print(f'✓ 周末效应因子已保存到 DuckDB 表: week_factor')
    
    print(f'  - 共 {len(monthly_factor)} 条记录（月度）')
    print(f'  - 覆盖 {monthly_factor["ts_code"].nunique()} 只股票')
    print(f'  - 时间范围: {monthly_factor["year_month"].min()} ~ {monthly_factor["year_month"].max()}')
    
    return monthly_factor
def compute_pricing_factors(stock_path, financial_path, out_dir, use_db=True):
    """
    计算定价因子（SMB、HML、UMD）
    依据 README 中定价因子计算部分的步骤：
    1. SMB: 上一月底的流动市值排名后三分之一股票组合的收益减去前三分之一股票组合的收益
    2. HML: 上一月底按账面市值比前三分之一股票组合的收益减去后三分之一股票组合的收益
    3. UMD: 上一月底按当月累积收益排名前三分之一股票组合的收益减去后三分之一股票组合的收益
    Args:
        stock_path: 股票市场数据路径 (兼容参数，use_db=True时不使用)
        financial_path: 股票财务数据路径 (兼容参数，use_db=True时不使用)
        out_dir: 输出目录 (Data/Processed/Derived_Factors, use_db=True时不使用)
        use_db: 是否使用DuckDB（推荐True）
    """    
    # 0. 检查是否已有数据（增量更新逻辑）
    out_path = os.path.join(out_dir, 'pricing_factors.csv')
    existing_dates = set()
    existing_data = None  # 已计算的 pricing_factors 数据（含 SMB/HML/UMD）
    
    try:
        if use_db:
            existing_data = db_utils.read_sql("SELECT * FROM pricing_factors")
        else:
            existing_data = pd.read_csv(out_path)
        existing_dates = set(existing_data['trade_date'].astype(int).unique())
        print(f'发现已有数据，包含 {len(existing_dates)} 个交易日')
        print(f'  最早日期: {min(existing_dates)}')
        print(f'  最晚日期: {max(existing_dates)}')
    except Exception as e:
        print(f'未找到已有pricing_factors数据（{e}），将重新计算全部数据')
        existing_dates = set()
    
    # 1. 读取股票数据
    # print('读取股票数据...')
    if use_db:
        # print('从DuckDB读取...')
        stock_data = db_utils.query_stock_bar(columns=['ts_code', 'trade_date', 'pct_chg'])
    else:
        stock_data = pd.read_csv(stock_path)
    stock_data = stock_data[['ts_code', 'trade_date', 'pct_chg']].copy()
    
    # 过滤出需要计算的日期（但要保留前几个月的数据用于计算月度汇总）
    need_full_data = True  # 标记是否需要读取完整数据
    if existing_dates:
        all_dates = set(stock_data['trade_date'].unique())
        new_dates = all_dates - existing_dates
        if not new_dates:
            print('所有数据已计算完成！')
            return existing_data
        print(f'需要计算 {len(new_dates)} 个新交易日')
        
        # 为了计算月度汇总，需要保留新日期所在月份及其前一个月的所有数据
        stock_data['temp_date'] = pd.to_datetime(stock_data['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
        min_new_date = min(new_dates)
        min_date_obj = pd.to_datetime(str(min_new_date), format='%Y%m%d')
        # 保留前2个月的数据（确保能获取上月数据）
        cutoff_date = (min_date_obj - pd.DateOffset(months=2)).replace(day=1)
        stock_data = stock_data[stock_data['temp_date'] >= cutoff_date].copy()
        stock_data = stock_data.drop(columns=['temp_date'])
        need_full_data = False
        # print(f'  保留从 {cutoff_date.strftime("%Y%m%d")} 开始的数据用于月度计算')
    
    # 2. 读取财务数据（包含市值和PB）
    # print('读取财务数据...')
    try:
        if use_db:
            financial_data = db_utils.query_daily_basic(columns=['ts_code', 'trade_date', 'total_mv', 'pb'])
        else:
            financial_data = pd.read_csv(financial_path)
        financial_data = financial_data[['ts_code', 'trade_date', 'total_mv', 'pb']].copy()
        
        # 如果是增量更新，保留同样的日期范围
        if not need_full_data:
            financial_data['temp_date'] = pd.to_datetime(financial_data['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
            financial_data = financial_data[financial_data['temp_date'] >= cutoff_date].copy()
            financial_data = financial_data.drop(columns=['temp_date'])
        
        # 合并股票数据和财务数据
        # print('合并数据...')
        df = pd.merge(stock_data, financial_data, on=['ts_code', 'trade_date'], how='inner')
    except Exception as e:
        print(f'警告：无法读取财务数据 {financial_path}，错误：{e}')
        return None
    
    if len(df) == 0:
        print('没有需要计算的新数据')
        return existing_data
    
    # print(f'数据总量：{len(df)} 行')
    print('开始计算新数据')
    # 3. 添加年月标识（用于分组，基于trade_date）
    # print('计算年月标识...')
    df['temp_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    df['year_month'] = df['temp_date'].dt.year * 100 + df['temp_date'].dt.month  # 更简单的月份标识
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    # 4. 计算每月累积收益（用于UMD因子）- 简化方法
    # print('计算月度累积收益...')
    # 直接累加取平均
    monthly_cum = df.groupby(['ts_code', 'year_month']).agg({
        'pct_chg': 'mean',  # 月度平均收益
        'trade_date': 'max',
        'total_mv': 'last',
        'pb': 'last'
    }).reset_index()
    monthly_cum['month_ret'] = monthly_cum['pct_chg']
    monthly_cum = monthly_cum[['ts_code', 'year_month', 'month_ret', 'total_mv', 'pb']].copy()
    
    # 5. 构建上月数据用于当月
    # print('构建上月数据映射...')
    monthly_cum['next_month'] = monthly_cum['year_month'] + 1
    # 处理跨年情况：12月的下个月是次年1月
    monthly_cum.loc[monthly_cum['year_month'] % 100 == 12, 'next_month'] = (monthly_cum.loc[monthly_cum['year_month'] % 100 == 12, 'year_month'] // 100 + 1) * 100 + 1
    
    monthly_cum_prev = monthly_cum[['ts_code', 'next_month', 'month_ret', 'total_mv', 'pb']].copy()
    monthly_cum_prev.columns = ['ts_code', 'year_month', 'prev_month_ret', 'prev_mv', 'prev_pb']
    
    # 合并上月数据到当日数据
    # print('合并上月数据到当日...')
    df = df.merge(monthly_cum_prev, on=['ts_code', 'year_month'], how='left')
    
    # 6. 按日期计算定价因子   
    # 定义单日计算函数
    def calc_daily_factors(trade_date, day_data):
        """计算单个交易日的定价因子"""
        # 过滤有效数据
        valid_data = day_data.dropna(subset=['pct_chg'])
        
        if len(valid_data) < 9:  # 至少需要9只股票才能分3组
            return None
        
        # SMB因子：基于上月底市值
        smb_factor = np.nan
        mv_data = valid_data.dropna(subset=['prev_mv'])
        if len(mv_data) >= 9:
            mv_data = mv_data.sort_values('prev_mv')
            n_stocks = len(mv_data)
            n_third = n_stocks // 3
            
            small_cap_ret = mv_data.iloc[:n_third]['pct_chg'].mean()
            large_cap_ret = mv_data.iloc[-n_third:]['pct_chg'].mean()
            smb_factor = small_cap_ret - large_cap_ret
        
        # HML因子：基于上月底PB（低PB为价值股，高PB为成长股）
        hml_factor = np.nan
        pb_data = valid_data.dropna(subset=['prev_pb'])
        if len(pb_data) >= 9:
            pb_data = pb_data.sort_values('prev_pb')
            n_stocks_pb = len(pb_data)
            n_third_pb = n_stocks_pb // 3
            
            value_ret = pb_data.iloc[:n_third_pb]['pct_chg'].mean()  # 低PB（价值股）
            growth_ret = pb_data.iloc[-n_third_pb:]['pct_chg'].mean()  # 高PB（成长股）
            hml_factor = value_ret - growth_ret
        
        # UMD因子：基于上月累积收益
        umd_factor = np.nan
        mom_data = valid_data.dropna(subset=['prev_month_ret'])
        if len(mom_data) >= 9:
            mom_data = mom_data.sort_values('prev_month_ret')
            n_stocks_mom = len(mom_data)
            n_third_mom = n_stocks_mom // 3
            
            winner_ret = mom_data.iloc[-n_third_mom:]['pct_chg'].mean()  # 高动量（赢家）
            loser_ret = mom_data.iloc[:n_third_mom]['pct_chg'].mean()   # 低动量（输家）
            umd_factor = winner_ret - loser_ret
        
        return {
            'trade_date': int(trade_date),
            'SMB': smb_factor,
            'HML': hml_factor,
            'UMD': umd_factor
        }
    
    # 按交易日分组
    daily_groups = list(df.groupby('trade_date'))
    total_days = len(daily_groups)
    # print(f'  共 {total_days} 个交易日，使用多线程加速...')
    
    # 并行计算
    pricing_factors = Parallel(n_jobs=-1, prefer="threads")(
        delayed(calc_daily_factors)(trade_date, day_data) 
        for trade_date, day_data in daily_groups
    )
    
    # 过滤掉None结果
    pricing_factors = [f for f in pricing_factors if f is not None]
        
    # 7. 输出结果
    result_df = pd.DataFrame(pricing_factors)
    
    # 如果是增量更新，只保留新日期的数据
    if existing_dates and len(result_df) > 0:
        result_df = result_df[result_df['trade_date'].isin(new_dates)].copy()
        print(f'  过滤后保留新日期数据: {len(result_df)} 条')
        
    # 合并新旧数据
    if existing_data is not None and len(result_df) > 0:
        print(f'合并新旧数据：已有 {len(existing_data)} 条，新增 {len(result_df)} 条')
        result_df = pd.concat([existing_data, result_df], ignore_index=True)
        # 去重，保留最新的数据
        result_df = result_df.drop_duplicates(subset=['trade_date'], keep='last')
    elif existing_data is not None:
        result_df = existing_data

    # 按日期降序排序（最新的在前）
    result_df = result_df.sort_values('trade_date', ascending=False).reset_index(drop=True)
    
    if use_db:
        db_utils.write_to_db(result_df, 'pricing_factors', if_exists='replace')
    else:
        # 保存三个因子到单独文件
        os.makedirs(out_dir, exist_ok=True)
        # 保存完整因子数据
        result_df.to_csv(out_path, index=False)
        print(f'定价因子计算完成：{out_path}')
    
    
    print(f'  - 共 {len(result_df)} 个交易日')
    print(f'  - SMB 有效值: {result_df["SMB"].notna().sum()}')
    print(f'  - HML 有效值: {result_df["HML"].notna().sum()}')
    print(f'  - UMD 有效值: {result_df["UMD"].notna().sum()}')  
    return result_df
