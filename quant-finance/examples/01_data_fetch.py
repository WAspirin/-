"""
量化金融 Day 1 - 股票数据获取与可视化

学习目标:
1. 使用 yfinance 获取美股数据
2. 使用 akshare 获取 A 股数据
3. 使用 matplotlib 绘制 K 线图
4. 计算基本统计指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 尝试导入 yfinance (美股)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("⚠️  yfinance 未安装，跳过美股数据获取")

# 尝试导入 akshare (A 股)
try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False
    print("⚠️  akshare 未安装，跳过 A 股数据获取")


def fetch_us_stock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取美股数据
    
    参数:
        symbol: 股票代码 (如 'AAPL', 'GOOGL')
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    返回:
        DataFrame with OHLCV data
    """
    if not HAS_YFINANCE:
        return None
    
    print(f"\n📈 获取美股 {symbol} 数据...")
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    
    if df.empty:
        print(f"  ✗ 未获取到数据")
        return None
    
    print(f"  ✓ 获取到 {len(df)} 条数据")
    print(f"  时间范围：{df.index[0].date()} ~ {df.index[-1].date()}")
    
    return df


def fetch_cn_stock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取 A 股数据
    
    参数:
        symbol: 股票代码 (如 '000001', '600519')
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    返回:
        DataFrame with OHLCV data
    """
    if not HAS_AKSHARE:
        return None
    
    print(f"\n📈 获取 A 股 {symbol} 数据...")
    try:
        # 获取日线数据
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', ''),
            adjust="qfq"  # 前复权
        )
        
        # 重命名列
        df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 
                      'Turnover', 'Amplitude', 'PctChg', 'Change', 'TurnoverRate']
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        print(f"  ✓ 获取到 {len(df)} 条数据")
        print(f"  时间范围：{df.index[0].date()} ~ {df.index[-1].date()}")
        
        return df
    except Exception as e:
        print(f"  ✗ 获取失败：{e}")
        return None


def plot_candlestick(df: pd.DataFrame, title: str = "股票价格"):
    """
    绘制简易 K 线图
    
    参数:
        df: 包含 OHLC 数据的 DataFrame
        title: 图表标题
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # 准备数据
    dates = df.index
    open_prices = df['Open']
    high_prices = df['High']
    low_prices = df['Low']
    close_prices = df['Close']
    
    # 绘制收盘价线
    ax1.plot(dates, close_prices, 'b-', linewidth=1.5, label='Close')
    ax1.fill_between(dates, open_prices, close_prices, 
                     where=(close_prices >= open_prices), 
                     interpolate=True, color='red', alpha=0.3, label='Gain')
    ax1.fill_between(dates, open_prices, close_prices, 
                     where=(close_prices < open_prices), 
                     interpolate=True, color='green', alpha=0.3, label='Loss')
    
    # 绘制最高价和最低价
    ax1.plot(dates, high_prices, 'gray', linewidth=0.5, alpha=0.5)
    ax1.plot(dates, low_prices, 'gray', linewidth=0.5, alpha=0.5)
    
    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 成交量
    colors = ['red' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'green' 
              for i in range(len(df))]
    ax2.bar(dates, df['Volume'], color=colors, alpha=0.5, width=1)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def calculate_statistics(df: pd.DataFrame, symbol: str):
    """
    计算基本统计指标
    
    参数:
        df: 价格数据
        symbol: 股票代码
    """
    print(f"\n📊 {symbol} 统计指标:")
    print("=" * 50)
    
    # 基本统计
    print(f"\n价格统计:")
    print(f"  最新收盘价：¥{df['Close'].iloc[-1]:.2f}")
    print(f"  最高价：¥{df['High'].max():.2f}")
    print(f"  最低价：¥{df['Low'].min():.2f}")
    print(f"  平均收盘价：¥{df['Close'].mean():.2f}")
    
    # 收益率
    df['Returns'] = df['Close'].pct_change()
    print(f"\n收益率统计:")
    print(f"  日均收益率：{df['Returns'].mean()*100:.3f}%")
    print(f"  日收益率标准差：{df['Returns'].std()*100:.3f}%")
    print(f"  最大单日涨幅：{df['Returns'].max()*100:.2f}%")
    print(f"  最大单日跌幅：{df['Returns'].min()*100:.2f}%")
    
    # 波动率
    volatility = df['Returns'].std() * np.sqrt(252)  # 年化波动率
    print(f"\n风险指标:")
    print(f"  年化波动率：{volatility*100:.2f}%")
    
    # 移动平均
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    print(f"\n移动平均:")
    print(f"  5 日均线：¥{df['MA5'].iloc[-1]:.2f}")
    print(f"  20 日均线：¥{df['MA20'].iloc[-1]:.2f}")
    
    print("=" * 50)


def main():
    """主函数"""
    print("=" * 60)
    print("量化金融 Day 1 - 股票数据获取与可视化")
    print("=" * 60)
    
    # 设置日期范围（最近 3 个月）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"\n📅 日期范围：{start_str} ~ {end_str}")
    
    # 获取美股数据
    if HAS_YFINANCE:
        us_df = fetch_us_stock('AAPL', start_str, end_str)
        if us_df is not None:
            calculate_statistics(us_df, 'AAPL')
            
            # 可视化
            fig1 = plot_candlestick(us_df, 'Apple Inc. (AAPL) - 90 天价格走势')
            output_path1 = "/root/.openclaw/workspace/quant-finance/examples/aapl_90days.png"
            fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
            print(f"\n✓ 图表已保存到：{output_path1}")
            plt.show()
    
    # 获取 A 股数据
    if HAS_AKSHARE:
        cn_df = fetch_cn_stock('000001', start_str, end_str)  # 平安银行
        if cn_df is not None:
            calculate_statistics(cn_df, '平安银行 (000001)')
            
            # 可视化
            fig2 = plot_candlestick(cn_df, '平安银行 (000001) - 90 天价格走势')
            output_path2 = "/root/.openclaw/workspace/quant-finance/examples/pay_90days.png"
            fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
            print(f"\n✓ 图表已保存到：{output_path2}")
            plt.show()
    
    print("\n" + "=" * 60)
    print("Day 1 学习完成！")
    print("=" * 60)
    print("\n📚 下一步:")
    print("  1. 学习 pandas 金融数据处理")
    print("  2. 计算技术指标 (MA, MACD, RSI)")
    print("  3. 实现双均线策略")
    print("\n💡 提示:")
    print("  - 如果库未安装，运行：pip install -r requirements.txt")
    print("  - 查看 learning-plan.md 了解完整学习计划")


if __name__ == "__main__":
    main()
