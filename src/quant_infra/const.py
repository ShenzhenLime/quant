# 存放常量，如指数代码、频率映射、常数

## 指数名称与代码转换
# 沪深300，中证500，中证800、中证1000，全市场
INDEX_NAME_TO_CODE = {
    '中证800': '000906.SH',  ##大中盘（沪深300+中证500）市值前800的股票
    '中证1000': '000852.SH', ## 小盘（中证1000）市值前801-1800的股票
    '全市场': 'all'
}

# pandas datetime频率映射字典
FREQ_MAP = {
    '日度': 'D',
    '周度': 'W',
    '月度': 'M',
}
## 计算单日IC时，最小的样本量
MIN_IC_SIZE = 10

## 默认的用3个标准差来计算极端值的范围
N_SIGMAS = 3

# 达到api限制后，等待的时间（秒）
LIMIT_SLEEP_SECONDS = 15

# 数据抓取并行进程数
DATA_FETCH_JOBS = 4

# 特质波动率相关窗口配置
RESID_REG_WINDOW = 252
SPEC_VOL_WINDOW = 20

# 数据库路径常量
DB_PATH = './Data/data.db'

# 基础信息存储路径常量
BASIC_INFO_PATH = 'Data/Metadata'

## 存储财务数据的更新时间（因为获取时间很久，而且财务数据频率较低，所以单独记录更新时间，避免每次运行都更新财务数据）
FETCH_LOG_PATH = f'{BASIC_INFO_PATH}/fetch_log.csv'
# 设置更新频率，单位为天。财务数据更新频率较低，所以设置为180天（半年）更新一次。
FINANCIAL_RENEW_DAYS = 180
# stock_basic 更新频率，一年更新一次
BASIC_RENEW_DAYS = 365

# 回测开始时间
START_DATE = '20160101'

# 行业信息更新频率，一年更新一次
INDUSTRY_RENEW_DAYS = 365
