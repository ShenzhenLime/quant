# quant_infra — 量化基础框架
**使用示例视频**：【耗时一个月，我搭建了最适合新手的完整量化框架。】 https://www.bilibili.com/video/BV1skXeBTEq3/?share_source=copy_web&vd_source=268eee2a1463962256f74e1052f0bbc1
---
## 📖 写在前面
> **熙熙攘攘，皆为利往，安于传道，能有几何？**
>
> 恰逢大一暑假，我看到油管上一个介绍 AI 量化交易的视频，便觉得量化很有意思。**让电脑交易，自己坐享其成，岂不美哉？** 然而初次的尝试却在各种环境配置、接口选择上接连碰壁，无奈之下，只好放弃。
>
> 一年之后，学院的《量化投资》课程给了我重操旧业的机会。在专业平台的助力下，我能花上几个周末的时间来学习研究，却未曾感到疲惫。也是在那时，通过几期视频，我便掌握了 SQL、Pandas的基础语法，也入门了机器学习。由此感慨，**量化并非深不可测，许多人只是被那些晦涩的技术名词挡在了门外。**
>
> 课程结束以后，我决定脱离云平台，尝试在本地搭建一套完整的量化环境。在这个过程中，又继续不断摸索，在数据下载、处理上掉了不少坑，最终整个环境才初具雏形。回想起这一路的不易，我便决定将搭建的这个环境在 GitHub 上免费地分享出来。我相信，这个项目能让更多想入门的朋友少走弯路，**从第一行成功返回的数据开始，感受量化的魅力** 。

---


`src/quant_infra` 是一套基于 **Tushare** 数据源、面向 A 股量化研究的基础设施库，提供从数据入库、因子预处理到因子评估、模拟交易的完整代码。所有行情数据以 **DuckDB** 本地数据库存储，避免对大 CSV 文件的反复读写；核心计算任务通过 **Joblib** 并行加速。

---

## 获取示例因子

本仓库使用 **Git Submodule** 管理因子库，克隆时请根据需要选择以下方式：

```bash
# 推荐：同时获取示例因子
git clone --recursive https://github.com/ShenzhenLime/quant.git

# 已克隆但尚未初始化子模块
git submodule update --init --recursive

# 不需要示例因子
git clone https://github.com/ShenzhenLime/quant.git
```

---
## 量化真经
- 打板策略不可信，实际交易时，根本不可能以开盘价或前一日收盘价成交。更专业的来说：**没有考虑滑点**。
- 从定价知识来看，逻辑简单的，通常没有好用的
- 在大盘股（估值低）和小盘股（业绩好）之间来回轮换的策略很多，聚宽社区里很常见，但本质上是风险的暴露

---
## 目前回测不错的因子
- spec_vol 特质波动率
---

## 模块总览

```
src/quant_infra/
├── const.py          # 全局常量
├── db_utils.py       # DuckDB 读写工具
├── get_data.py       # 行情 & 基本面数据多进程获取（Tushare）
├── factor_calc.py    # 定价因子 & 残差收益计算
├── factor_analyze.py # 因子评估 & 可视化
└── trade.py          # 真实交易模拟回测（含手续费 & 滑点）
```

---

## 模块详解

### `const.py` — 全局常量

| 常量 | 说明 |
|------|------|
| `INDEX_NAME_TO_CODE` | 指数名称 → Tushare 代码映射（中证800、中证1000、全市场） |
| `FREQ_MAP` | 中文频率 → Pandas 频率缩写（日度/周度/月度 → D/W/M） |

---

### `db_utils.py` — 数据库工具

以 DuckDB 文件（`Data/data.db`）作为统一数据仓库，封装了三个核心函数：

| 函数 | 说明 |
|------|------|
| `init_db()` | 初始化并返回数据库连接，自动创建目录 |
| `read_sql(query)` | 执行任意 SQL，返回 DataFrame |
| `write_to_db(df, table_name, save_mode)` | 将 DataFrame 写入指定表，支持 `replace`（覆盖）和 `append`（追加）两种模式 |

---

### `get_data.py` — 数据获取

通过 **Tushare Pro** 接口下载数据并持久化到 DuckDB。Token 从项目根目录 `.env` 中的 `TS_TOKEN` 读取，**不硬编码**于代码中。

#### 增量更新机制

`get_dates_todo(table_name)` 是所有下载函数的核心调度器，会自动完成：

1. 读取本地交易日历（`Data/Metadata/trade_day.csv`），如日历过期则自动更新；
2. 查询 DuckDB 中目标表的最新日期；
3. 返回"尚未入库"的交易日列表，**避免重复下载**。

#### 数据获取函数

| 函数 | DuckDB 表 / 本地文件 | 说明 |
|------|------------|------|
| `get_stock_data_by_date()` | `stock_bar` | 全市场股票日频行情（open/close/pct_chg 等），自动过滤涨跌幅异常（>35%）数据，内置重试机制 |
| `get_daily_basic()` | `daily_basic` | 每日指标（PB、PE、总市值、换手率等） |
| `get_index_data(index_code)` | `index_data` | 指定指数的日频行情 |
| `get_ins(index_code)` | `Data/Metadata/{code}_ins.csv` | 获取指数最近一个月的成分股列表 |
| `get_trade(start, end)` | `Data/Metadata/trade_day.csv` | 更新交易日历 |

所有日频数据下载均通过 `joblib.Parallel` + `tqdm` 进行**4 进程并行**，并内置限流重试（遇到 Tushare 频率限制时自动等待）。

---

### `factor_calc.py` — 因子计算

#### 1. Fama-French 四因子（`compute_pricing_factors()`）

基于全市场日线数据，按交易日并行计算 MKT / SMB / HML / UMD 四个定价因子，结果写入 `pricing_factors` 表。

| 因子 | 计算逻辑 |
|------|----------|
| **MKT** | 当日全市场股票平均收益率 |
| **SMB** | 按**上月末流通市值**排序，后三分之一（小盘）减去前三分之一（大盘）的收益 |
| **HML** | 按**上月末 PB** 排序，前三分之一（低PB/价值）减去后三分之一（高PB/成长）的收益 |
| **UMD** | 按**上月累积收益率**排序，后三分之一（强势）减去前三分之一（弱势）的收益 |

> 所有排序基准均使用**上月末**已知数据，杜绝未来数据泄漏。

#### 2. 股票残差收益率（`calc_resid()`）

对每只股票按交易日进行**滚动回归**：对每个交易日 `t`，使用最近 `RESID_REG_WINDOW` 个交易日的收益与四因子数据回归当期 Beta，并仅保留 `t` 当天的残差，结果写入 `stock_resids` 表。残差收益率可作为剥离系统性风险后的纯 Alpha 信号。

整个流程支持**增量更新**：补算新日期时会自动向前回看足够长的历史窗口，以保证滚动回归可用。

#### 3. 去极值（`winsorize(series, n=3)`）

按 n 倍标准差对序列上下缩尾，用于压制异常值对因子分析的干扰。

---

### `factor_analyze.py` — 因子评估

#### `evaluate_factor(factor_table, fac_freq)`

因子评估主函数，对因子在**多样本 × 多调仓频率**的所有组合下进行并行回测，输出评价指标。

**样本范围**：中证800 / 中证1000 / 全市场  
**调仓频率**：根据因子自身频率自动过滤（如月频因子不会评估日度调仓）

每种组合的评估流程：
1. 若非日频，将日度收益和因子值按周期聚合；
2. 合并因子与下期收益，计算 **Rank IC**（Spearman 相关系数）；
3. 按当期因子值分 **10 组**，追踪各组日度持仓收益；
4. 输出 **IC / IR / Sharpe**（多空组合年化）及各分组累计收益。

汇总指标保存至 `factor_mining/{factor_table}/output/summary.csv`，日度多空收益序列写入 DuckDB 的 `{factor_table}_daily_ls` 表。


#### `group_plot(sample, freq, line, factor_table, mode)`

从 DuckDB 读取指定样本 & 频率的日度收益序列，绘制**累计净值曲线**并保存为 PNG。

| `mode` 参数 | 读取的 DuckDB 表 | 可选 `line` |
|-------------|----------------|------------|
| `'evaluate'`（默认）| `{factor_table}_daily_ls` | `ls_ret` / `long` / `short` |
| `'trade'` | `{factor_table}_trade_daily_ret` | `long` |
| `'pathway'` | `{factor_table}_pathway_daily_ls` | `ls_ret` / `long` / `short` |

---

### `trade.py` — 交易模拟

#### `simulate_trade(factor_table, trade_freq, ...)`

真实交易模拟回测，评价因子对头部股票的选股能力。相比 `factor_analyze` 中的等权分组回测，本模块更贴近实盘，具体增加了：

- **手续费 + 滑点**：双边计算，默认单边万 2.5 手续费 + 0.1% 滑点；
- **板块过滤**：可排除创业板（300 开头）、科创板（68 开头）、北交所（4/8 开头）；
- **高价股过滤**：默认过滤收盘价 > 100 元的股票。

**参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `factor_table` | — | 因子表名（DuckDB） |
| `trade_freq` | — | 调仓频率（如 `'月度'`） |
| `bench_index` | `'000002.SH'` | 基准指数 |
| `sample` | `'全市场'` | 回测样本 |
| `n_top` | `5` | 每期持仓股票数 |
| `is_ascending` | `False` | 因子排序方向，`False` 表示因子越大越好 |
| `commission_rate` | `2.5` | 单边手续费（万分比） |
| `slippage_rate` | `0.1` | 单边滑点（百分比） |
| `price_max` | `100.0` | 高价股过滤上限（元） |
| `filter_boards` | `('创业板','科创板','北交所')` | 排除的板块，传空元组不过滤 |

**输出**

- `factor_mining/{factor_table}/output/trade_holdings.csv`：每期换仓时持仓股票列表；
- DuckDB 表 `{factor_table}_trade_daily_ret`：日度净值序列（含 `long` 及 `bench_ret` 列），供 `group_plot(mode='trade')` 使用。

#### `compute_portfolio_daily_ret(stk_df, holdings, ...)`

`simulate_trade` 的内部函数，按换仓周期合并日度等权收益，在每期首日扣除**双边交易成本**（仅对实际发生换手的仓位计费）。通过 `joblib` 多线程并行处理各期，最终返回完整日度 `long` 收益序列。

---

## 数据流示意

```
Tushare API
    │
    ▼  get_data.py（增量下载）
┌─────────────────────────┐
│  DuckDB (Data/data.db)  │
│  ├── stock_bar          │  ← 日频行情
│  ├── daily_basic        │  ← 每日指标
│  ├── index_data         │  ← 指数行情
│  ├── pricing_factors    │  ← MKT/SMB/HML/UMD
│  └── stock_resids       │  ← 滚动回归残差
└─────────────────────────┘
    │
    ▼  factor_analyze.py
因子评估结果
  ├── summary.csv              ← IC / IR / Sharpe / 分组收益
  ├── {factor}_daily_ls        ← 日度多空收益（DuckDB）
  └── *_curve.png              ← 净值曲线图
    │
    ▼  trade.py
交易模拟结果
  ├── trade_holdings.csv       ← 每期换仓持仓记录
  └── {factor}_trade_daily_ret ← 日度净值序列（DuckDB）
```

---

## 环境配置

**Python >= 3.12**，依赖见 `pyproject.toml`：

```
pandas / numpy / joblib / matplotlib / duckdb / tqdm / tushare / scipy 
```

安装：

```bash
pip install -e .
```

在项目根目录创建 `.env`，配置 Tushare Token（需要2000积分及以上）：

```bash
TS_TOKEN=你的token
```

