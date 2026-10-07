# Inter IIT Bootcamp – Quant '26: BTC & ETH Trading Strategies

Rule-based breakout strategies for BTC/USDT and ETH/USDT on 15-minute data, 1 Jan 2021 – 31 Dec 2025. Each account starts with 100,000 USDT.

## Files
| File | Purpose |
|---|---|
| `bitcoin_crash_strategy.py` | BTC strategy and backtest engine |
| `eth_final_strategy.py` | ETH strategy and backtest engine |
| `BTCUSDT_15m.csv`, `ETHUSDT_15m.csv` | Input data |
| `Quant26_BTC_ETH_Report.pdf` | Full research report |

## Strategy
Both strategies combine a **daily trend filter** with a **4-hour breakout entry**. Stops are ATR-based, size is set by volatility, and drawdown controls cut exposure. Signals use completed candles only, and orders fill at the next tradable 15-minute open.

### BTC (slower, more selective)
- **Long:** daily close above 40-day SMA; 4h close above previous 36-bar high; volatility rank ≤ 0.90; weekdays only.
  Stop 4 ATR; profit lock after +1.25 ATR; 54-bar exit channel.
- **Short:** daily close below 99% of 75-day SMA (SMA lower than 30 days earlier); break of previous 72-bar low; Hurst ≥ 0.45.
  Stop 3 ATR; lock at +1.5 ATR; 4 ATR trail; exit on 18-bar channel or loss of bearish regime.
- **Risk:** long cap 100%, short cap 80%; drawdown thresholds 15% / 26%; 1-day cooldown; 60-day max hold.

### ETH (faster)
- **Long:** daily close above 35-day SMA; 4h close above previous 10-bar high; ADX ≥ 20; volatility rank ≤ 0.90; weekends allowed.
  Stop 2 ATR; lock after +2.5 ATR; 36-bar exit channel.
- **Short:** daily close below 35-day SMA (itself below its value 10 days earlier); 4h close below previous 36-bar low.
  Stop 2.5 ATR; 18-bar exit channel.
- **Risk:** long cap 100%, short cap 65%; drawdown thresholds 25% / 30%; 30-minute cooldown; 60-day max hold.

## Costs and Accounting
- 0.15% of notional per transaction (fee + slippage).
- 5% annual carry on short exposure.
- Zero-volume bars cannot trade. The final position is closed at the last open.
- Sharpe: daily returns, zero risk-free rate, √365.

## Results (costs and carry already deducted)
| Metric | BTC L/S | BTC Long | ETH L/S | ETH Long |
|---|---|---|---|---|
| Final equity (USDT) | 832,999 | 575,382 | 1,254,582 | 587,590 |
| Total return | 733.00% | 475.38% | 1,154.58% | 487.59% |
| CAGR | 52.81% | 41.91% | 65.86% | 42.51% |
| Sharpe | 1.72 | 1.54 | 1.46 | 1.16 |
| Sortino | 3.00 | 2.63 | 2.68 | 2.15 |
| Max drawdown | 23.59% | 23.59% | 24.51% | 28.69% |
| Closed trades | 64 | 45 | 122 | 65 |
| Win rate | 82.81% | 86.67% | 57.38% | 43.08% |
| Quarters beating buy-and-hold | 11 / 20 | 11 / 20 | 10 / 20 | 10 / 20 |
| Buy-and-hold return | 202.25% | 202.25% | 302.54% | 302.54% |

BTC beats buy-and-hold in more than half of quarters. ETH reaches exactly half, so that target is not met.

## Robustness (synthetic paths, 1,000 per test)
| Test | BTC median CAGR | ETH median CAGR |
|---|---|---|
| Block Monte Carlo | 29.73% | 31.41% |
| HMM | 6.70% | -6.32% |
| Fair random walk | -4.41% | -6.35% |

The strategies hold up when real trend structure is kept. They weaken when trends are scrambled, since they depend on trends.

## Limitations
- Both accounts are flat much of the time; longest underwater spells approach 300 days.
- Costs are a fixed assumption, not an order-book model.
- Short results assume borrow and carry, and do not prove availability.
- Performance on hidden competition segments is not established.

## What did not work
AR, Ornstein-Uhlenbeck mean reversion, indicator-only strategies, cross-sectional correlation and Random Forest ML (too slow to run and tune). We also reviewed four papers (sentiment-based, deep Q-network, control-chart and ML-ensemble approaches). See Section 4 of the report for the reasons.

## References
- Garcia & Schweitzer (2015): https://pmc.ncbi.nlm.nih.gov/articles/PMC4593685/
- Otabek & Choi (2024): https://www.nature.com/articles/s41598-024-51408-w
- Yeganeh et al. (2025): https://www.sciencedirect.com/science/article/pii/S0952197624018104
- Sebastião & Godinho (2021): https://link.springer.com/article/10.1186/s40854-020-00217-x
