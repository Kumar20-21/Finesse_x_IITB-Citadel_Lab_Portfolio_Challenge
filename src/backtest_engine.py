"""
Backtest engine: composite factor scoring, inverse-volatility weighting with a cap, and
quarterly rebalancing with leftover-cash redeployment.
"""
import pandas as pd
import numpy as np

INITIAL_CAPITAL = 1_00_00_000
TXN_COST = 0.001
N_HOLDINGS = 10
MOM_LOOKBACK = 252
MOM_SKIP = 21
VOL_LOOKBACK = 90


def composite_scores(close, as_of_date, mom_weight, quality_weight):
    """
    Computes the composite factor score (Momentum, Low-Vol, Quality) for every eligible
    stock as of as_of_date. Returns (scores sorted descending, trailing volatility).
    """
    idx_loc = close.index.get_loc(as_of_date)
    if idx_loc < MOM_LOOKBACK:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    px = close.iloc[:idx_loc + 1]

    p_skip = px.iloc[-1 - MOM_SKIP]
    p_lb = px.iloc[-1 - MOM_LOOKBACK]
    mom = (p_skip / p_lb) - 1.0
    z_mom = (mom - mom.mean()) / mom.std()
    mom_valid_mask = px.iloc[-MOM_LOOKBACK:].notna().sum() >= MOM_LOOKBACK * 0.95

    vol = px.pct_change().iloc[-VOL_LOOKBACK:].std()
    lowvol = -vol

    valid = z_mom.notna() & lowvol.notna() & mom_valid_mask

    window = px.iloc[-MOM_LOOKBACK:]
    log_px = np.log(window)
    x_series = pd.Series(np.arange(len(window)), index=window.index, dtype=float)
    r = log_px.corrwith(x_series)
    z_quality = (r**2 - (r**2).mean()) / (r**2).std()
    valid = valid & z_quality.notna()

    z_mom = z_mom[valid]
    lowvol = lowvol[valid]
    vol = vol[valid]
    z_lv = (lowvol - lowvol.mean()) / lowvol.std()
    z_quality = z_quality[valid]

    composite = ((1 - quality_weight) * (mom_weight * z_mom + (1 - mom_weight) * z_lv)
                 + quality_weight * z_quality)

    return composite.sort_values(ascending=False), vol


def capped_weights(raw_affinity, cap, eps=0.1):
    """Inverse-volatility weights: shift affinities positive, normalise, then iteratively
    cap at `cap`, redistributing excess proportionally to uncapped names."""
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


def run_backtest(close, sma200, start, end, *, mom_weight=0.5, weight_cap=0.15, quality_weight=0.5,
                  redeploy_leftover=True, redeploy_cap=None, initial_capital=INITIAL_CAPITAL,
                  n_holdings=N_HOLDINGS, invvol_eps=0.1, dd_breaker=None, dd_breaker_frac=0.5):
    """
    Runs the quarterly-rebalanced backtest from start to end and returns
    (equity_curve, trade_log, closed_trades). A selected stock already trading below its own
    200-day average on the rebalance date itself is not bought that quarter.

    redeploy_cap: caps each name's weight during the leftover-cash redeployment pass (None =
    uncapped, the submitted behaviour; weight_cap re-enforces the initial-selection cap instead).
    dd_breaker, dd_breaker_frac: if dd_breaker is set (e.g. 0.08), sells dd_breaker_frac of every
    holding, once per quarter, the first time portfolio value falls dd_breaker below its
    post-rebalance peak for that quarter. None (default) disables this, the submitted behaviour.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    sim_days = close.index[(close.index >= start) & (close.index <= end)]

    q_starts = pd.date_range(start.replace(month=1, day=1), end, freq='QS')
    rebal_dates = []
    for qs in q_starts:
        future = sim_days[sim_days >= max(qs, start)]
        if len(future) > 0:
            rebal_dates.append(future[0])
    rebal_set = set(rebal_dates)

    cash = initial_capital
    shares = pd.Series(0.0, index=close.columns)
    avg_entry = pd.Series(np.nan, index=close.columns)
    current_target_list = set()
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
        if date not in rebal_set:
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
            continue

        scores, vol = composite_scores(close, date, mom_weight, quality_weight)
        target_list = list(scores.index[:n_holdings])
        current_target_list = set(target_list)

        port_val = cash + (shares * px_today.reindex(shares.index).fillna(0)).sum()
        inv = 1.0 / vol.loc[target_list]
        w = capped_weights(inv, cap=weight_cap, eps=invvol_eps)
        target_alloc = {t: w[t] * port_val for t in target_list}

        for t in target_list:
            p_t = px_today.get(t, np.nan)
            s_t = sma200.loc[date, t] if t in sma200.columns else np.nan
            if pd.notna(p_t) and pd.notna(s_t) and p_t < s_t:
                target_alloc[t] = 0.0

        drop = [t for t in shares.index if shares[t] > 0 and t not in current_target_list]
        for t in drop:
            p = px_today.get(t, np.nan)
            if pd.isna(p) or shares[t] <= 0:
                continue
            execute(t, date, 'SELL', shares[t], p)

        # Simulate the main trim/buy pass and the leftover-cash redeployment pass against a
        # local cash/share balance first, accumulating one signed net delta per name, so each
        # name executes as exactly one order.
        net_delta = {}
        sim_cash = cash
        sim_shares = {t: shares[t] for t in target_list}
        trimmed = set()
        for t in target_list:
            p = px_today.get(t, np.nan)
            if pd.isna(p) or p <= 0:
                continue
            diff_val = target_alloc[t] - shares[t] * p
            if diff_val < 0:
                n = np.floor(min(shares[t], -diff_val / p))
                if n >= 1:
                    net_delta[t] = net_delta.get(t, 0) - n
                    sim_shares[t] -= n
                    sim_cash += n * p * (1 - TXN_COST)
                trimmed.add(t)
            elif diff_val > 0:
                n = np.floor(diff_val / p)
                if n >= 1 and n * p * (1 + TXN_COST) <= sim_cash:
                    net_delta[t] = net_delta.get(t, 0) + n
                    sim_shares[t] += n
                    sim_cash -= n * p * (1 + TXN_COST)

        if redeploy_leftover:
            # Top up names that got funded (excluding ones trimmed this quarter), proportional
            # to their own target weights, with cash left after the main pass.
            for _ in range(5):
                funded = [t for t in target_list if t not in trimmed and sim_shares.get(t, 0) > 0]
                if not funded or sim_cash <= 0:
                    break
                if redeploy_cap is not None:
                    port_val_now = sim_cash + sum(sim_shares[t] * px_today.get(t, np.nan) for t in funded)
                    cur_val = pd.Series({t: sim_shares[t] * px_today.get(t, np.nan) for t in funded})
                    headroom = (redeploy_cap * port_val_now - cur_val).clip(lower=0)
                    headroom = headroom[headroom > 0]
                    if headroom.empty:
                        break
                    room_names = headroom.index
                else:
                    headroom = None
                    room_names = funded
                w_room = w.loc[room_names]
                w_room = w_room / w_room.sum()
                bought_any = False
                for t in room_names:
                    p = px_today.get(t, np.nan)
                    if pd.isna(p) or p <= 0:
                        continue
                    extra_cash = w_room[t] * sim_cash
                    if headroom is not None:
                        extra_cash = min(extra_cash, headroom[t])
                    n = np.floor(extra_cash / p)
                    if n >= 1 and n * p * (1 + TXN_COST) <= sim_cash:
                        net_delta[t] = net_delta.get(t, 0) + n
                        sim_shares[t] += n
                        sim_cash -= n * p * (1 + TXN_COST)
                        bought_any = True
                if not bought_any:
                    break

        for t, n in net_delta.items():
            if n < 0:
                p = px_today.get(t, np.nan)
                if pd.notna(p) and p > 0:
                    execute(t, date, 'SELL', -n, p)
        for t, n in net_delta.items():
            if n > 0:
                p = px_today.get(t, np.nan)
                if pd.notna(p) and p > 0 and n * p <= cash:
                    execute(t, date, 'BUY', n, p)

        port_val = cash + (shares * px_today.reindex(shares.index).fillna(0)).sum()
        quarter_peak = port_val
        breaker_triggered = False
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
