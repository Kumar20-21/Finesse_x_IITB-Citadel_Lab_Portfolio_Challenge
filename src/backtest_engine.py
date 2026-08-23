"""
Core backtest engine: stock selection, position sizing, rebalancing, and the daily
trend-filter risk overlay described in the report's Methodology section.

The final submitted strategy uses one fixed configuration of this engine (see
run_backtest.py). The extra parameters below (mom_ensemble, downside_vol, dd_breaker,
sector_cap, regime_bench) exist because the validation scripts in validation/ use this
same engine to reproduce the alternative designs that were tested and rejected during
development (see the report's Methodology / Limitations sections for why).
"""
import pandas as pd
import numpy as np

INITIAL_CAPITAL = 1_00_00_000
TXN_COST = 0.001
N_HOLDINGS = 10
MOM_LOOKBACK = 252
MOM_SKIP = 21
VOL_LOOKBACK = 90
TREND_WINDOW = 200


MOM_ENSEMBLE_LOOKBACKS = [63, 126, 252]  # 3M / 6M / 12M, each skipping the most recent month


ADV_LOOKBACK = 20


def composite_scores(close, as_of_date, mom_weight, mom_ensemble=False,
                      quality_weight=0.0, downside_vol=False,
                      volume=None, min_adv_pctile=None):
    """
    volume, min_adv_pctile: optional liquidity screen. If both given, stocks whose trailing
      20-day average daily traded value (price x volume) falls below the min_adv_pctile
      percentile of that day's eligible universe are excluded before scoring, so illiquid
      names with stale-price (artificially low) measured volatility can't be favoured by the
      Low-Vol factor or over-sized by inverse-vol weighting.
    """
    idx_loc = close.index.get_loc(as_of_date)
    if idx_loc < MOM_LOOKBACK:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    px = close.iloc[:idx_loc + 1]

    if mom_ensemble:
        z_list = []
        for lb in MOM_ENSEMBLE_LOOKBACKS:
            p_skip_ = px.iloc[-1 - MOM_SKIP]
            p_lb_ = px.iloc[-1 - lb]
            m = (p_skip_ / p_lb_) - 1.0
            z_list.append((m - m.mean()) / m.std())
        z_mom = pd.concat(z_list, axis=1).mean(axis=1)
        mom_valid_mask = px.iloc[-MOM_LOOKBACK:].notna().sum() >= MOM_LOOKBACK * 0.95
    else:
        p_skip = px.iloc[-1 - MOM_SKIP]
        p_lb = px.iloc[-1 - MOM_LOOKBACK]
        mom = (p_skip / p_lb) - 1.0
        z_mom = (mom - mom.mean()) / mom.std()
        mom_valid_mask = px.iloc[-MOM_LOOKBACK:].notna().sum() >= MOM_LOOKBACK * 0.95

    rets_full = px.pct_change().iloc[-VOL_LOOKBACK:]
    if downside_vol:
        # semi-deviation: only downside days count as "risk", so crash-prone names get
        # penalised more than a symmetric-vol measure would penalise a smooth grinder.
        vol = np.sqrt((rets_full.clip(upper=0) ** 2).mean())
    else:
        vol = rets_full.std()
    lowvol = -vol

    valid = z_mom.notna() & lowvol.notna() & mom_valid_mask

    if volume is not None and min_adv_pctile is not None:
        vol_px = volume.iloc[:idx_loc + 1]
        adv = (px.iloc[-ADV_LOOKBACK:] * vol_px.iloc[-ADV_LOOKBACK:]).mean()
        liquid = adv >= adv[valid].quantile(min_adv_pctile)
        valid = valid & liquid.reindex(valid.index).fillna(False)

    if quality_weight > 0:
        window = px.iloc[-MOM_LOOKBACK:]
        log_px = np.log(window)
        x_series = pd.Series(np.arange(len(window)), index=window.index, dtype=float)
        r = log_px.corrwith(x_series)
        z_quality = (r**2 - (r**2).mean()) / (r**2).std()
        valid = valid & z_quality.notna()
    else:
        z_quality = None

    z_mom = z_mom[valid]
    lowvol = lowvol[valid]
    vol = vol[valid]
    z_lv = (lowvol - lowvol.mean()) / lowvol.std()

    if quality_weight > 0:
        z_quality = z_quality[valid]
        composite = ((1 - quality_weight) * (mom_weight * z_mom + (1 - mom_weight) * z_lv)
                      + quality_weight * z_quality)
    else:
        composite = mom_weight * z_mom + (1 - mom_weight) * z_lv

    return composite.sort_values(ascending=False), vol


def select_top_n_with_sector_cap(scores, industry_map, n, sector_cap=None):
    if sector_cap is None:
        return list(scores.index[:n])
    picked, counts = [], {}
    for t in scores.index:
        ind = industry_map.get(t, "Unknown")
        if counts.get(ind, 0) >= sector_cap:
            continue
        picked.append(t)
        counts[ind] = counts.get(ind, 0) + 1
        if len(picked) == n:
            break
    if len(picked) < n:  # relax cap only if universe can't fill 10 slots otherwise
        for t in scores.index:
            if t not in picked:
                picked.append(t)
                if len(picked) == n:
                    break
    return picked


def capped_weights(raw_affinity, cap, eps=0.1):
    shifted = raw_affinity - raw_affinity.min() + eps
    w = shifted / shifted.sum()
    for _ in range(10):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if w[under].sum() <= 0:
            break
        w[under] = w[under] + excess * (w[under] / w[under].sum())
    return w


def run_backtest(close, sma200, start, end, *, mom_weight=0.5, weighting="equal",
                  weight_cap=0.20, reentry=False, initial_capital=INITIAL_CAPITAL,
                  dd_breaker=None, dd_breaker_frac=0.5, mom_ensemble=False,
                  sector_cap=None, industry_map=None, quality_weight=0.0,
                  downside_vol=False, regime_bench=None, regime_defensive_frac=0.5,
                  volume=None, min_adv_pctile=None):
    """
    weighting: "equal" | "score" | "invvol"
    reentry: allow mid-quarter re-entry once price reclaims its 200-DMA
    dd_breaker: if set (e.g. 0.08), de-risk dd_breaker_frac of every holding, once per
      quarter, the first time portfolio value falls dd_breaker fraction below its
      post-rebalance peak for that quarter.
    mom_ensemble: average z-scored 3M/6M/12M momentum instead of a single 12-1M lookback.
    sector_cap: max stocks per industry among the top-10 (requires industry_map: ticker -> industry).
    quality_weight: blend in a trend-smoothness (R-squared of log-price trend) factor, to avoid
      picking spiky/parabolic movers that are more prone to sharp reversal.
    downside_vol: use downside semi-deviation instead of full std for the low-vol factor and
      inverse-vol sizing, so crash risk is penalised more directly than symmetric volatility.
    regime_bench: optional benchmark close-price Series; if given, at each rebalance the target
      allocation is scaled by regime_defensive_frac whenever the benchmark itself is below its
      own 200-DMA (a market-level, not stock-level, risk-off overlay).
    volume, min_adv_pctile: optional liquidity screen, see composite_scores().
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    trading_days = close.index
    sim_days = trading_days[(trading_days >= start) & (trading_days <= end)]

    regime_sma = regime_bench.rolling(200, min_periods=200).mean() if regime_bench is not None else None

    q_starts = pd.date_range(start.replace(month=1, day=1), end, freq='QS')
    rebal_dates = []
    for qs in q_starts:
        future = sim_days[sim_days >= max(qs, start)]
        if len(future) > 0:
            rebal_dates.append(future[0])
    rebal_set = set(sorted(set(rebal_dates)))

    cash = initial_capital
    shares = pd.Series(0.0, index=close.columns)
    avg_entry = pd.Series(np.nan, index=close.columns)
    current_target_list = set()
    target_alloc = {}
    exited_pending_reentry = set()
    quarter_peak = initial_capital
    breaker_triggered = False

    equity_curve, trade_log, closed_trades = [], [], []

    def execute(ticker, date, side, n_shares, price):
        nonlocal cash
        notional = n_shares * price
        cost = notional * TXN_COST
        if side == 'BUY':
            cash -= (notional + cost)
            prev_shares = shares[ticker]
            prev_val = prev_shares * (avg_entry[ticker] if prev_shares > 0 else 0)
            new_shares = prev_shares + n_shares
            avg_entry[ticker] = (prev_val + notional) / new_shares
            shares[ticker] = new_shares
        else:
            cash += (notional - cost)
            shares[ticker] -= n_shares
            if shares[ticker] <= 1e-9:
                entry_p = avg_entry[ticker]
                closed_trades.append({
                    'Ticker': ticker, 'ExitDate': date, 'EntryPrice': entry_p,
                    'ExitPrice': price, 'ReturnPct': (price / entry_p - 1.0) * 100,
                    'Profitable': price > entry_p
                })
                shares[ticker] = 0.0
                avg_entry[ticker] = np.nan
        trade_log.append({'Date': date, 'Ticker': ticker, 'Side': side,
                           'Shares': n_shares, 'Price': price, 'Notional': notional, 'Cost': cost})

    for date in sim_days:
        px_today = close.loc[date]

        if date in rebal_set:
            scores, vol = composite_scores(close, date, mom_weight, mom_ensemble=mom_ensemble,
                                            quality_weight=quality_weight, downside_vol=downside_vol,
                                            volume=volume, min_adv_pctile=min_adv_pctile)
            target_list = select_top_n_with_sector_cap(scores, industry_map or {}, N_HOLDINGS, sector_cap=sector_cap)
            current_target_list = set(target_list)
            exited_pending_reentry = set()

            port_val = cash + (shares * px_today.reindex(shares.index).fillna(0)).sum()

            if weighting == "equal":
                w = pd.Series(1.0 / N_HOLDINGS, index=target_list)
            elif weighting == "score":
                w = capped_weights(scores.loc[target_list], cap=weight_cap)
            elif weighting == "invvol":
                inv = 1.0 / vol.loc[target_list]
                w = capped_weights(inv, cap=weight_cap)
            else:
                raise ValueError(weighting)

            exposure_scale = 1.0
            if regime_sma is not None and date in regime_bench.index and pd.notna(regime_sma.get(date, np.nan)):
                if regime_bench[date] < regime_sma[date]:
                    exposure_scale = regime_defensive_frac

            target_alloc = {t: w[t] * port_val * exposure_scale for t in target_list}

            drop = [t for t in shares.index if shares[t] > 0 and t not in current_target_list]
            for t in drop:
                p = px_today.get(t, np.nan)
                if pd.isna(p) or shares[t] <= 0:
                    continue
                execute(t, date, 'SELL', shares[t], p)

            # Sells before buys: trim over-target positions first so their freed cash is
            # available to fund under-target positions in the same rebalance, regardless of
            # scan order. Without this, a lower-priority buy can be starved of cash that a
            # later-in-list sell would otherwise have released.
            diffs = {}
            for t in target_list:
                p = px_today.get(t, np.nan)
                if pd.isna(p) or p <= 0:
                    continue
                diffs[t] = (target_alloc[t] - shares[t] * p, p)

            for t, (diff_val, p) in diffs.items():
                if diff_val < 0:
                    n = np.floor(min(shares[t], -diff_val / p))
                    if n >= 1:
                        execute(t, date, 'SELL', n, p)

            for t, (diff_val, p) in diffs.items():
                if diff_val > 0:
                    n = np.floor(diff_val / p)
                    if n >= 1 and n * p <= cash:
                        execute(t, date, 'BUY', n, p)

            quarter_peak = cash + (shares * px_today.reindex(shares.index).fillna(0)).sum()
            breaker_triggered = False

        held = [t for t in shares.index if shares[t] > 0]
        for t in held:
            p = px_today.get(t, np.nan)
            s = sma200.loc[date, t] if t in sma200.columns else np.nan
            if pd.isna(p) or pd.isna(s):
                continue
            if p < s:
                execute(t, date, 'SELL', shares[t], p)
                if reentry and t in current_target_list:
                    exited_pending_reentry.add(t)

        if reentry:
            # Deterministic order: iterating a Python set directly is hash-seed dependent
            # (varies across process runs), which silently made cash-constrained re-entry
            # ties non-reproducible. Sort by target allocation (highest-conviction first)
            # so any cash-limited tie-break is both deterministic and economically motivated.
            reentry_order = sorted(exited_pending_reentry, key=lambda t: -target_alloc.get(t, 0))
            for t in reentry_order:
                if shares[t] > 0:
                    exited_pending_reentry.discard(t)
                    continue
                p = px_today.get(t, np.nan)
                s = sma200.loc[date, t] if t in sma200.columns else np.nan
                if pd.isna(p) or pd.isna(s):
                    continue
                if p >= s:
                    alloc = target_alloc.get(t, 0)
                    n = np.floor(alloc / p)
                    if n >= 1 and n * p <= cash:
                        execute(t, date, 'BUY', n, p)
                    exited_pending_reentry.discard(t)

        port_val = cash + (shares * px_today.reindex(shares.index).fillna(0)).sum()
        quarter_peak = max(quarter_peak, port_val)

        if dd_breaker is not None and not breaker_triggered and shares.gt(0).any():
            if port_val < quarter_peak * (1 - dd_breaker):
                for t in [tt for tt in shares.index if shares[tt] > 0]:
                    p = px_today.get(t, np.nan)
                    if pd.isna(p):
                        continue
                    n = np.floor(shares[t] * dd_breaker_frac)
                    if n >= 1:
                        execute(t, date, 'SELL', n, p)
                breaker_triggered = True
                port_val = cash + (shares * px_today.reindex(shares.index).fillna(0)).sum()

        equity_curve.append({'Date': date, 'PortfolioValue': port_val, 'Cash': cash})

    eq = pd.DataFrame(equity_curve).set_index('Date')
    tl = pd.DataFrame(trade_log)
    ct = pd.DataFrame(closed_trades)
    return eq, tl, ct


def summarize(eq, tl, ct, initial_capital=INITIAL_CAPITAL, label=""):
    end_val = eq['PortfolioValue'].iloc[-1]
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (end_val / initial_capital) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    abs_ret = end_val / initial_capital - 1
    rets = eq['PortfolioValue'].pct_change().dropna()
    mdd = (eq['PortfolioValue'] / eq['PortfolioValue'].cummax() - 1).min()
    sharpe = cagr / (rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
    if len(ct):
        win_rate = ct['Profitable'].mean()
        wins = ct.loc[ct['Profitable'], 'ReturnPct']
        losses = ct.loc[~ct['Profitable'], 'ReturnPct']
        gl = wins.mean() / abs(losses.mean()) if len(losses) else np.nan
    else:
        win_rate = gl = np.nan
    turnover = tl['Notional'].sum() / initial_capital if len(tl) else 0
    cost = tl['Cost'].sum() if len(tl) else 0
    return {
        'label': label, 'end_val': end_val, 'net_pnl': end_val - initial_capital,
        'abs_ret_pct': abs_ret * 100, 'cagr_pct': cagr * 100, 'mdd_pct': mdd * 100,
        'sharpe': sharpe, 'win_rate_pct': win_rate * 100 if pd.notna(win_rate) else np.nan,
        'gain_loss': gl, 'n_orders': len(tl), 'n_closed_trades': len(ct),
        'turnover_x': turnover, 'txn_cost': cost,
    }


def print_summary(s):
    print(f"--- {s['label']} ---")
    print(f"Final Value: Rs {s['end_val']:,.0f}   Net PnL: Rs {s['net_pnl']:,.0f}")
    print(f"Abs Return: {s['abs_ret_pct']:.2f}%   CAGR: {s['cagr_pct']:.2f}%   MDD: {s['mdd_pct']:.2f}%   Sharpe: {s['sharpe']:.2f}")
    print(f"Win rate: {s['win_rate_pct']:.2f}%   Gain/Loss: {s['gain_loss']:.2f}   Orders: {s['n_orders']}   ClosedTrades: {s['n_closed_trades']}")
    print(f"Turnover: {s['turnover_x']:.1f}x   TxnCost: Rs {s['txn_cost']:,.0f}")
    print()
