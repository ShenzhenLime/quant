# DuckDB 数据库完整指南

## 📖 目录
- [快速开始](#快速开始)
- [核心功能](#核心功能)
- [API参考](#api参考)
- [实战示例](#实战示例)
- [性能优化](#性能优化)
- [故障排查](#故障排查)


### 核心优势

1. **列式存储**: 只读取需要的列，不像CSV需要读取整行
2. **自动索引**: 主键自动索引，查询速度快
3. **SQL查询**: 支持复杂的筛选、聚合、连接操作
4. **增量更新**: 只需下载新数据，不用重复下载全量
5. **并行处理**: 自动利用多核CPU加速查询

## 快速开始

### 1️⃣ 首次使用（已完成）

转换脚本已执行完成，数据已导入数据库：

```bash
python convert_csv_to_duckdb.py
```

**转换结果**:
- ✅ stock_bar: 10,264,780 行，5,703只股票
- ✅ daily_basic: 9,775,273 行，5,702只股票  
- ✅ index_data: 5,109 行

**数据库位置**: `./Data/data.db` (846MB)

### 2️⃣ 在代码中使用

#### 方式A: 使用现有流程（推荐）

项目已完全适配，直接运行即可：

```python
# main.py
python main.py  # 默认使用DuckDB
```

所有函数已支持 `use_db=True` 参数（默认值）。

#### 方式B: 直接查询数据

```python
import db_utils

# 查询股票数据
df = db_utils.query_stock_bar(start_date='20240101')

# 查询每日指标
df = db_utils.query_daily_basic(start_date='20240101')

# 查询指数数据
df = db_utils.query_index_data()
```

#### 方式C: SQL查询

```python
import db_utils

# 自定义SQL
df = db_utils.read_sql("""
    SELECT ts_code, trade_date, close, pct_chg
    FROM stock_bar
    WHERE trade_date >= '20240101'
        AND pct_chg > 5.0
    ORDER BY pct_chg DESC
    LIMIT 100
""")
```

### 3️⃣ 查看数据统计

```python
import db_utils

# 查看表统计
stats = db_utils.get_table_stats('stock_bar')
print(stats)

# 输出示例:
# table_name  row_count  unique_codes  min_date  max_date
# stock_bar   10264780   5703         20160104  20260114
```

## 核心功能

### 数据库结构

```
./Data/data.db
├── stock_bar      # 股票日线数据
│   ├── ts_code    # 股票代码
│   ├── trade_date # 交易日期
│   ├── open       # 开盘价
│   ├── high       # 最高价
│   ├── low        # 最低价
│   ├── close      # 收盘价
│   ├── pct_chg    # 涨跌幅
│   └── ...
├── daily_basic    # 每日基本信息
│   ├── ts_code    # 股票代码
│   ├── trade_date # 交易日期
│   ├── total_mv   # 总市值
│   ├── pb         # 市净率
│   ├── pe         # 市盈率
│   └── ...
└── index_data     # 指数数据
    ├── ts_code    # 指数代码
    ├── trade_date # 交易日期
    ├── close      # 收盘点位
    └── pct_chg    # 涨跌幅
```

### 已适配的模块

#### ✅ main.py - 数据获取
- `get_stock_data_by_date(output, use_db=True)` - 获取股票数据
- `get_daily_basic(output, use_db=True)` - 获取每日指标
- `get_index_data(index_code)` - 获取指数数据
- **支持增量更新**: 自动检测已有数据

#### ✅ factor_construct.py - 因子计算
- `compute_raw_factor(..., use_db=True)` - 计算原始因子
- `compute_momentum(..., use_db=True)` - 计算动量因子
- `merge_factors(..., use_db=True)` - 合并因子
- `compute_pricing_factors(..., use_db=True)` - 计算定价因子

#### ✅ factor_analyze.py - 因子分析
- `evaluate_factor(..., use_db=True)` - 因子评估
- `_weekly_returns(..., use_db=True)` - 周度收益计算

## API参考

### db_utils 模块完整API

#### 查询函数

**1. query_stock_bar() - 查询股票日线数据**

```python
df = db_utils.query_stock_bar(
    start_date=None,      # 开始日期，格式: '20240101'
    end_date=None,        # 结束日期
    ts_codes=None         # 股票代码列表，如 ['000001.SZ', '600000.SH']
)
```

**示例**:
```python
# 查询所有数据
df = db_utils.query_stock_bar()

# 查询指定日期范围
df = db_utils.query_stock_bar(
    start_date='20240101',
    end_date='20241231'
)

# 查询指定股票
df = db_utils.query_stock_bar(
    ts_codes=['000001.SZ', '600000.SH']
)

# 组合条件
df = db_utils.query_stock_bar(
    start_date='20240101',
    ts_codes=['000001.SZ']
)
```

**2. query_daily_basic() - 查询每日基本信息**

```python
df = db_utils.query_daily_basic(
    start_date=None,
    end_date=None,
    ts_codes=None
)
```

包含字段: `total_mv`(总市值), `pb`(市净率), `pe`(市盈率), `turnover_rate`(换手率)等

**3. query_index_data() - 查询指数数据**

```python
df = db_utils.query_index_data(
    ts_code=None,         # 指数代码，如 '000852.SH'
    start_date=None,
    end_date=None
)
```

**4. read_sql() - 执行自定义SQL**

```python
df = db_utils.read_sql(sql_query)
```

**示例**:
```python
# 复杂查询
df = db_utils.read_sql("""
    SELECT 
        ts_code,
        SUBSTR(trade_date, 1, 6) as month,
        AVG(close) as avg_close,
        AVG(pct_chg) as avg_return
    FROM stock_bar
    WHERE trade_date >= '20240101'
    GROUP BY ts_code, month
    ORDER BY month, avg_return DESC
""")
```

#### 写入函数

**5. write_to_db() - 写入DataFrame到数据库**

```python
db_utils.write_to_db(
    df,                   # pandas DataFrame
    table_name,          # 表名
    if_exists='replace'  # 'replace' 或 'append'
)
```

**6. csv_to_db() - CSV导入数据库**

```python
db_utils.csv_to_db(
    csv_path,            # CSV文件路径
    table_name,          # 目标表名
    if_exists='replace'  # 'replace' 或 'append'
)
```

**7. append_to_db() - 追加数据**

```python
db_utils.append_to_db(csv_path, table_name)
```

#### 工具函数

**8. get_table_stats() - 获取表统计**

```python
stats = db_utils.get_table_stats(table_name)
# 返回: 行数、唯一代码数、日期范围
```

**9. export_db_to_csv() - 导出为CSV**

```python
db_utils.export_db_to_csv(table_name, output_path)
```

**10. init_db() - 初始化数据库连接**

```python
conn = db_utils.init_db()
# 使用完记得关闭
conn.close()
```

## 实战示例

### 示例1: 查询近期涨幅最大的股票

```python
import db_utils

# 查询最近交易日涨幅超过5%的股票
df = db_utils.read_sql("""
    SELECT ts_code, trade_date, close, pct_chg
    FROM stock_bar
    WHERE trade_date = (SELECT MAX(trade_date) FROM stock_bar)
        AND pct_chg > 5.0
    ORDER BY pct_chg DESC
    LIMIT 20
""")

print(f"找到 {len(df)} 只涨停股票")
print(df)
```

### 示例2: 计算月度平均收益

```python
import db_utils

df = db_utils.read_sql("""
    SELECT 
        ts_code,
        SUBSTR(trade_date, 1, 6) as month,
        AVG(pct_chg) as avg_monthly_return,
        SUM(pct_chg) as cum_monthly_return,
        COUNT(*) as trading_days
    FROM stock_bar
    WHERE trade_date >= '20240101'
    GROUP BY ts_code, month
    HAVING COUNT(*) >= 15  -- 至少15个交易日
    ORDER BY avg_monthly_return DESC
""")

# 找出表现最好的股票
top_performers = df.groupby('ts_code')['avg_monthly_return'].mean().nlargest(10)
print("年度最佳股票:", top_performers)
```

### 示例3: 查询高市值股票

```python
import db_utils

# 查询最新交易日市值最大的50只股票
df = db_utils.read_sql("""
    SELECT 
        b.ts_code,
        b.trade_date,
        b.close,
        d.total_mv,
        d.pe,
        d.pb
    FROM stock_bar b
    INNER JOIN daily_basic d 
        ON b.ts_code = d.ts_code 
        AND b.trade_date = d.trade_date
    WHERE b.trade_date = (SELECT MAX(trade_date) FROM stock_bar)
    ORDER BY d.total_mv DESC
    LIMIT 50
""")

print(f"A股市值前50:")
print(df[['ts_code', 'total_mv', 'pe', 'pb']])
```

### 示例4: 计算股票波动率

```python
import db_utils
import numpy as np

# 查询数据
df = db_utils.query_stock_bar(
    start_date='20240101',
    ts_codes=['000001.SZ', '600000.SH']
)

# 计算波动率
volatility = df.groupby('ts_code')['pct_chg'].std()
print("年化波动率:", volatility * np.sqrt(252))
```

### 示例5: 行业轮动分析

```python
import db_utils

# 计算不同市值分组的平均收益
df = db_utils.read_sql("""
    SELECT 
        CASE 
            WHEN d.total_mv < 5000000 THEN 'Small'
            WHEN d.total_mv < 20000000 THEN 'Mid'
            ELSE 'Large'
        END as cap_group,
        b.trade_date,
        AVG(b.pct_chg) as avg_return
    FROM stock_bar b
    INNER JOIN daily_basic d 
        ON b.ts_code = d.ts_code 
        AND b.trade_date = d.trade_date
    WHERE b.trade_date >= '20240101'
    GROUP BY cap_group, b.trade_date
    ORDER BY b.trade_date, cap_group
""")

# 透视表展示
pivot = df.pivot(index='trade_date', columns='cap_group', values='avg_return')
print(pivot.tail(10))
```

### 示例6: 增量更新数据

```python
# 在main.py中
def main():
    use_db = True
    
    # 自动检测已有数据，只下载新的
    print("\n增量更新股票数据...")
    get_stock_data_by_date(
        f'{MARKET_PATH}/stock_bar.csv',
        use_db=use_db
    )
    
    print("\n增量更新财务数据...")
    get_daily_basic(
        f'{FINANCIALS_PATH}/daily_basic.csv',
        use_db=use_db
    )
```

### 示例7: 完整分析流程

```python
import db_utils

# 1. 查询数据
df = db_utils.query_stock_bar(start_date='20240101')
df_basic = db_utils.query_daily_basic(start_date='20240101')

# 2. 合并数据
merged = df.merge(df_basic, on=['ts_code', 'trade_date'])

# 3. 计算因子
merged['momentum_20'] = merged.groupby('ts_code')['pct_chg'].rolling(20).sum().values
merged['size_factor'] = np.log(merged['total_mv'])

# 4. 因子分析
# ... 进行因子分析

# 5. 结果保存
db_utils.write_to_db(merged, 'analysis_result', if_exists='replace')
```

## 性能优化

### 1. 使用日期过滤

```python
# ✅ 好 - 只查询需要的数据
df = db_utils.query_stock_bar(start_date='20240101', end_date='20241231')

# ❌ 不好 - 查询全部数据再筛选
df = db_utils.query_stock_bar()
df = df[df['trade_date'] >= '20240101']
```

### 2. 使用股票代码过滤

```python
# ✅ 好 - 在数据库层面筛选
df = db_utils.query_stock_bar(ts_codes=['000001.SZ', '600000.SH'])

# ❌ 不好 - 查询全部再筛选
df = db_utils.query_stock_bar()
df = df[df['ts_code'].isin(['000001.SZ', '600000.SH'])]
```

### 3. 只选择需要的列

```python
# ✅ 好 - 只查询需要的列
df = db_utils.read_sql("""
    SELECT ts_code, trade_date, close, pct_chg
    FROM stock_bar
    WHERE trade_date >= '20240101'
""")

# ❌ 不好 - SELECT * 查询所有列
df = db_utils.read_sql("""
    SELECT *
    FROM stock_bar
    WHERE trade_date >= '20240101'
""")
```

### 4. 在数据库中完成聚合

```python
# ✅ 好 - 数据库层面聚合
df = db_utils.read_sql("""
    SELECT ts_code, AVG(pct_chg) as avg_return
    FROM stock_bar
    WHERE trade_date >= '20240101'
    GROUP BY ts_code
""")

# ❌ 不好 - Python层面聚合
df = db_utils.query_stock_bar(start_date='20240101')
df = df.groupby('ts_code')['pct_chg'].mean()
```

### 5. 使用索引

表已自动创建主键索引：
- `stock_bar`: (ts_code, trade_date)
- `daily_basic`: (ts_code, trade_date)
- `index_data`: (ts_code, trade_date)

查询时尽量使用这些字段作为条件。

## 故障排查

### 问题1: DuckDB未安装

**症状**: `ModuleNotFoundError: No module named 'duckdb'`

**解决**:
```bash
pip install duckdb
```

### 问题2: 数据库文件损坏

**症状**: 查询时报错或返回异常数据

**解决**:
```bash
# 删除数据库文件
rm ./Data/data.db

# 重新转换
python convert_csv_to_duckdb.py
```

### 问题3: 查询速度慢

**可能原因**:
1. 没有使用日期过滤
2. SELECT * 查询所有列
3. 在Python中做聚合而不是在SQL中

**解决**: 参考[性能优化](#性能优化)章节

### 问题4: 内存不足

**症状**: `MemoryError` 或系统变慢

**解决**:
1. 使用日期范围查询减少数据量
2. 分批处理数据
3. 使用SQL聚合减少返回数据量

```python
# 分批处理
dates = pd.date_range('20240101', '20241231', freq='M')
for i in range(len(dates)-1):
    start = dates[i].strftime('%Y%m%d')
    end = dates[i+1].strftime('%Y%m%d')
    df = db_utils.query_stock_bar(start_date=start, end_date=end)
    # 处理这批数据
```

### 问题5: 数据不一致

**症状**: 查询结果与预期不符

**解决**:
```python
# 检查表统计
import db_utils
stats = db_utils.get_table_stats('stock_bar')
print(stats)

# 查看最新日期
df = db_utils.read_sql("SELECT MAX(trade_date) FROM stock_bar")
print("最新日期:", df.iloc[0, 0])

# 对比CSV和数据库数据量
import pandas as pd
csv_count = len(pd.read_csv('./Data/Stock_data/Market/stock_bar.csv'))
db_count = db_utils.read_sql("SELECT COUNT(*) FROM stock_bar").iloc[0, 0]
print(f"CSV: {csv_count}, DB: {db_count}")
```

## 维护建议

### 定期备份

```bash
# 备份数据库文件
cp ./Data/data.db ./Data/data.db.backup_$(date +%Y%m%d)

# 或导出为CSV
python -c "
import db_utils
db_utils.export_db_to_csv('stock_bar', './backup/stock_bar_backup.csv')
db_utils.export_db_to_csv('daily_basic', './backup/daily_basic_backup.csv')
"
```

### 查看数据库信息

```python
import db_utils

# 查看所有表
conn = db_utils.init_db()
tables = conn.execute("SHOW TABLES").fetchall()
print("数据库中的表:", [t[0] for t in tables])

# 查看表结构
schema = conn.execute("DESCRIBE stock_bar").fetchall()
for col in schema:
    print(f"{col[0]}: {col[1]}")

# 查看数据库大小
import os
size_mb = os.path.getsize('./Data/data.db') / 1024 / 1024
print(f"数据库大小: {size_mb:.2f} MB")

conn.close()
```

### 数据清理

```python
import db_utils

# 删除旧数据（例如3年前的）
conn = db_utils.init_db()
conn.execute("""
    DELETE FROM stock_bar 
    WHERE trade_date < '20210101'
""")
conn.close()

# 或重建表（更快）
# 1. 导出最近数据
# 2. 删除表
# 3. 重新创建并导入
```

## 常见问题

**Q: 为什么选择DuckDB而不是SQLite或MySQL？**

A: 
- **vs SQLite**: DuckDB是列式存储，在分析查询（聚合、扫描）上比SQLite快10-100倍
- **vs MySQL**: DuckDB是嵌入式数据库，无需服务器，且对OLAP（分析）工作负载优化更好
- **vs Pandas**: 大数据量时DuckDB内存占用更少，查询更快

**Q: 数据库会自动更新吗？**

A: 是的，调用数据获取函数时会自动检测已有数据，只下载新的交易日数据。

**Q: 可以同时使用CSV和数据库吗？**

A: 可以，通过 `use_db` 参数控制：
```python
# 使用数据库
compute_raw_factor(..., use_db=True)

# 使用CSV
compute_raw_factor(..., use_db=False)
```

**Q: 如何添加新表？**

A: 
```python
import db_utils

# 方式1: 从DataFrame创建
df = pd.DataFrame(...)
db_utils.write_to_db(df, 'new_table', if_exists='replace')

# 方式2: 从CSV创建
db_utils.csv_to_db('data.csv', 'new_table')

# 方式3: SQL创建
conn = db_utils.init_db()
conn.execute("""
    CREATE TABLE new_table (
        id INTEGER,
        name VARCHAR,
        value DOUBLE
    )
""")
conn.close()
```

**Q: DuckDB支持哪些SQL功能？**

A: DuckDB支持完整的SQL标准，包括：
- 窗口函数（OVER, PARTITION BY）
- 通用表表达式（WITH ... AS）
- 子查询、连接（JOIN）
- 聚合函数（AVG, SUM, COUNT等）
- 字符串函数、日期函数
- CASE WHEN 条件表达式
- [完整文档](https://duckdb.org/docs/sql/introduction)

**Q: 如何查看执行计划？**

A:
```python
import db_utils

conn = db_utils.init_db()
plan = conn.execute("""
    EXPLAIN SELECT * FROM stock_bar 
    WHERE trade_date >= '20240101'
""").fetchall()
print(plan)
conn.close()
```

**Q: 支持并发访问吗？**

A: DuckDB支持多读单写。同一时间可以有多个读操作，但只能有一个写操作。对于本项目的使用场景（主要是读取和分析）完全足够。

## 学习资源

### DuckDB官方文档
- [官方网站](https://duckdb.org/)
- [SQL参考](https://duckdb.org/docs/sql/introduction)
- [Python API](https://duckdb.org/docs/api/python/overview)

### 推荐教程
- [DuckDB vs Pandas性能对比](https://duckdb.org/why_duckdb)
- [数据分析最佳实践](https://duckdb.org/docs/guides/overview)

### 测试和验证
```bash
# 运行测试脚本
python test_duckdb.py
```

## 总结

### ✅ 已完成
- ✅ 数据迁移完成（2.2GB → 846MB）
- ✅ 所有代码已适配DuckDB
- ✅ 支持增量更新
- ✅ 完全向后兼容
- ✅ 性能大幅提升

### 🎯 使用建议
1. **日常使用**: 直接运行 `python main.py`，默认使用DuckDB
2. **数据查询**: 使用 `db_utils` 模块的便捷函数
3. **复杂分析**: 使用SQL查询，在数据库层面完成聚合
4. **定期备份**: 备份 `./Data/data.db` 文件

### 📚 进阶使用
- 学习SQL可以解锁更多强大功能
- 查看 `db_utils.py` 了解实现细节
- 参考[实战示例](#实战示例)进行复杂分析

---

**项目已全面升级到DuckDB！** 🚀

如有问题，请查看[故障排查](#故障排查)或运行 `python test_duckdb.py` 进行诊断。
