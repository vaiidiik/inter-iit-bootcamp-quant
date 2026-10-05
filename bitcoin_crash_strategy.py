from dataclasses import dataclass, asdict, replace
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
try:
    from numba import njit
except ImportError:
    def njit(*a, **k): return lambda f:f
INITIAL=100000.

@dataclass(frozen=True)
class Config:
    rule: str='4h'
    offset_hours: int=0
    trend_mode: int=0
    momentum_days: int=40
    entry_threshold: float=.66
    exit_threshold: float=0.
    vol_fast_days: int=7
    vol_slow_days: int=30
    regime_days: int=75
    breakout: int=30
    exit_bars: int=9
    stop_atr: float=2.5
    lock_trigger: float=1.5
    lock_profit: float=.5
    trail_atr: float=0.
    vol_rank_max: float=.8
    vol_target: float=1.2
    weekends: bool=False
    long_cap: float=1.
    short_hysteresis: float=0.
    short_require_long_bear: bool=False
    short_mode: int=1
    short_cap: float=.65
    short_breakout: int=24
    short_exit: int=6
    short_stop: float=2.5
    short_lock: float=1.5
    short_vol_rank: float=1.01
    short_regime_days: int=0
    short_slope_days: int=5
    short_regime_exit: bool=True
    short_trail_atr: float=0.
    short_lock_profit: float=-1.
    short_vol_target: float=0.
    short_signal_mode: int=0
    short_rally_ema: int=20
    short_shock_atr: float=0.
    short_shock_bars: int=6
    only_shorts: bool=False
    regime_exit: bool=False
    hurst_min: float=0.
    long_adx_min: float=0.
    short_adx_min: float=0.
    short_hurst_min: float=0.
    channel_close: bool=False
    pullback_rsi: float=0.
    dd_soft: float=.99
    dd_hard: float=.999
    cooldown_bars: int=4
    max_hold_bars: int=5760
    cost: float=.0015
    carry: float=.05
    delay: int=0
    long_short: bool=True

class Features:
    def __init__(self,d):
        self.d=d;self.cache={};self.day=resample(d,'1D')
    def get(self,p):
        per=int(pd.Timedelta('1D')/pd.Timedelta(p.rule))
        key=(p.rule,p.offset_hours,p.vol_fast_days,p.vol_slow_days)
        if key not in self.cache:
            b=resample(self.d,p.rule,pd.Timedelta(hours=p.offset_hours));per=int(pd.Timedelta('1D')/pd.Timedelta(p.rule))
            lr=np.log(b.close).diff()
            v=lr.rolling(p.vol_slow_days*per).std()*np.sqrt(365*per)
            b['vol']=pd.concat([lr.rolling(p.vol_fast_days*per).std()*np.sqrt(365*per),v],axis=1).max(axis=1)
            b['rank']=v.rolling(90*per,min_periods=30*per).rank(pct=True)

            logp=np.log(b.close)
            v1=logp.diff().rolling(90*per,min_periods=30*per).var()
            v16=logp.diff(16).rolling(90*per,min_periods=30*per).var()
            b['hurst']=(np.log(v16/v1)/(2*np.log(16))).clip(0,1)
            up=b.high.diff();dn=-b.low.diff();plus=up.where((up>dn)&(up>0),0.).ewm(alpha=1/14,adjust=False).mean();minus=dn.where((dn>up)&(dn>0),0.).ewm(alpha=1/14,adjust=False).mean()
            b['adx']=(100*(plus-minus).abs()/(plus+minus)).ewm(alpha=1/14,adjust=False,min_periods=28).mean()
            self.cache[key]=b
        b=self.cache[key];ma=self.day.close.rolling(p.regime_days).mean()
        bull=((self.day.close>ma)&self.day.complete).reindex(b.index,method='ffill',fill_value=False)
        sma=self.day.close.rolling(p.short_regime_days or p.regime_days).mean()
        bear=((self.day.close<sma*(1-p.short_hysteresis))&(sma<sma.shift(p.short_slope_days))&self.day.complete).reindex(b.index,method='ffill',fill_value=False)
        if p.short_shock_atr>0:
            fast=b.close.ewm(span=12,adjust=False).mean()
            shock=(b.close-b.close.shift(p.short_shock_bars)<-p.short_shock_atr*b.atr.shift(p.short_shock_bars))&(b.close<fast)&(fast<fast.shift(3))
            bear= bear|shock
        en=((b.close>(b.close if p.channel_close else b.high).shift().rolling(p.breakout).max())|((b.rsi<p.pullback_rsi)&(p.pullback_rsi>0)))&bull&(b['rank']<=p.vol_rank_max)&(b.hurst>=p.hurst_min)&(b.adx>=p.long_adx_min)
        if not p.weekends:en&=b.index.dayofweek<5
        sn=(b.close<b.low.shift().rolling(p.short_breakout).min())&bear&(b['rank']<=p.short_vol_rank)&(b.hurst>=p.short_hurst_min)&(b.adx>=p.short_adx_min)
        if p.short_signal_mode==1:sn=(b.close<b.close.ewm(span=p.short_breakout,adjust=False).mean())&(b.close<b.close.shift(3))&bear&(b['rank']<=p.short_vol_rank)&(b.hurst>=p.short_hurst_min)&(b.adx>=p.short_adx_min)
        if p.short_signal_mode in (2,3):
            rally_ema=b.close.ewm(span=p.short_rally_ema,adjust=False).mean()
            failed_rally=(b.high>=rally_ema.shift())&(b.close<rally_ema)&(b.close<b.open)
            cross=(b.close<rally_ema)&(b.close.shift()>=rally_ema.shift())
            sn=(failed_rally if p.short_signal_mode==2 else cross)&bear&(b['rank']<=p.short_vol_rank)&(b.hurst>=p.short_hurst_min)&(b.adx>=p.short_adx_min)
        if p.short_mode==2:sn&=b.close<b.close.ewm(span=20,adjust=False).mean()
        if p.short_require_long_bear:sn &= ~bull
        if p.short_mode==0 or not p.long_short:sn&=False
        lx=b.close<b.low.shift().rolling(p.exit_bars).min()
        sx=(b.close>b.high.shift().rolling(p.short_exit).max())|(~bear & p.short_regime_exit)
        if p.regime_exit:lx|=~bull
        if p.trend_mode:
            score=pd.concat([np.sign(b.close/b.close.shift(int(p.momentum_days*k*per))-1) for k in (.5,1.,2.)],axis=1).mean(axis=1)
            en=(score>=p.entry_threshold)&bull&(b['rank']<=p.vol_rank_max)&(b.hurst>=p.hurst_min)&(b.adx>=p.long_adx_min)
            sn=(score<=-p.entry_threshold)&bear&(b.hurst>=p.short_hurst_min)&(b.adx>=p.short_adx_min)&(p.long_short)&(p.short_mode!=0)
            lx=score<=p.exit_threshold; sx=(score>=-p.exit_threshold)|~bear
            if p.trend_mode==2: en=bull;sn=bear&(p.long_short);lx=~bull;sx=~bear
            if not p.weekends: en &= b.index.dayofweek<5
        if p.only_shorts:en &= False
        frame=pd.DataFrame(dict(signal_close=b.close,en=en,sn=sn,lx=lx,sx=sx,valid=b.complete,atr=b.atr,size=pd.Series(np.where(en,p.vol_target,p.short_vol_target or p.vol_target),index=b.index).div(b.vol).clip(upper=1),hurst=b.hurst,vol=b.vol,vol_rank=b['rank'],adx=b.adx,bull=bull,bear=bear,short_break=(b.close<b.low.shift().rolling(p.short_breakout).min()),short_below_ma=(self.day.close<sma).reindex(b.index,method='ffill',fill_value=False),short_slope=(sma<sma.shift(p.short_slope_days)).reindex(b.index,method='ffill',fill_value=False)))
        aligned=frame.reindex(self.d.index)
        if p.delay:aligned=aligned.shift(p.delay)

        self.update_clock=aligned.valid.eq(True).to_numpy(bool)
        self.update_prices=aligned.signal_close.fillna(0).to_numpy(float)


        events=np.flatnonzero(aligned.valid.eq(True).to_numpy())
        eligible=np.flatnonzero(self.d.tradable.to_numpy())
        destinations=np.searchsorted(eligible,events)
        keep=destinations<len(eligible)
        events=events[keep]; destinations=eligible[destinations[keep]]

        unique=~pd.Index(destinations).duplicated(keep='last')
        queued=aligned.iloc[events[unique]].copy();queued.index=self.d.index[destinations[unique]]
        aligned=queued.reindex(self.d.index)
        flags=[aligned[k].eq(True).to_numpy(bool) for k in ['en','sn','lx','sx','valid']]
        vals=[pd.to_numeric(aligned[k]).fillna(0).to_numpy(float) for k in ['atr','size']]
        return (*flags,*vals),frame

@njit(cache=True)
def simulate(o,h,l,c,liquid,en,sn,lx,sx,valid,atr,size,start,end,initial,cost,carry,stop_atr,lock_trigger,lock_profit,trail_atr,short_cap,short_stop,short_lock,ddsoft,ddhard,cooldown,maxhold,update_clock=None,short_trail_atr=0.,short_lock_profit=-1.,long_cap=1.,update_prices=None):
    n=len(c);eq=np.full(n,np.nan);exp=np.zeros(n);cp=np.full(n,np.nan);qp=np.zeros(n);dd=np.zeros(n);cy=np.zeros(n)
    stop_path=np.full(n,np.nan)
    orders=np.zeros((2*n+4,8));trades=np.zeros((n+2,7));oc=tc=0
    cash=initial;q=0.;stop=entry=ea=0.;peak=initial;born=start;until=start;ep_base=initial;fees=ecarry=totalcarry=0.;locked=False
    maxgross=0.;gaps=intrabr=0;recovering=False;risk_pending=False
    last=end-1

    for i in range(start,end):
        if i>start and q<0:
            charge=abs(q)*c[i-1]*carry/35040.;cash-=charge;totalcarry+=charge;ecarry+=charge
        pre=cash+q*o[i];gross=abs(q)*o[i]/max(pre,1e-12);maxgross=max(maxgross,gross)
        if gross>1+1e-9:gaps+=1

        localdd=1-pre/max(peak,1e-12)
        if localdd<=ddsoft:recovering=False
        risktrip=localdd>ddhard and not recovering
        if risktrip:recovering=True;risk_pending=q!=0;until=max(until,i+288)
        reason=0;close=False
        if q!=0 and liquid[i]:
            if (q>0 and o[i]<=stop) or (q<0 and o[i]>=stop):close=True;reason=1
            elif valid[i] and ((q>0 and lx[i]) or (q<0 and sx[i])):close=True;reason=0
            elif valid[i] and ((q>0 and sn[i]) or (q<0 and en[i])):close=True;reason=8
            elif i-born>=maxhold:close=True;reason=4
            elif risk_pending:close=True;reason=5
            elif gross>=.995 and q<0:close=True;reason=6
        if i==last and q!=0 and liquid[i]:close=True;reason=7
        if close:
            risk_pending=False
            side=np.sign(q);delta=-q;fee=abs(q)*o[i]*cost;cash-=delta*o[i]+fee;fees+=fee
            orders[oc]=np.array([i,delta,o[i],fee,reason,q,0.,cash]);oc+=1
            trades[tc]=np.array([born,i,side,cash-ep_base,fees,ecarry,(i-born)/96.]);tc+=1;q=0.
            if reason in (1,4,5,6):until=max(until,i+cooldown)
        if q==0 and i<last and i>=until and valid[i] and liquid[i] and atr[i]>0:
            side=1. if en[i] else (-1. if sn[i] else 0.)
            if side!=0:
                fraction=min(long_cap,size[i]) if side>0 else min(short_cap,size[i])
                localdd=1-cash/max(peak,1e-12)
                if localdd>ddsoft:fraction*=.5
                if localdd>ddhard:fraction*=.5
                ep_base=cash;fees=ecarry=0.;born=i;entry=o[i];ea=atr[i];locked=False
                q=side*fraction*cash/(o[i]*(1+fraction*cost));fee=abs(q)*o[i]*cost;cash-=q*o[i]+fee;fees+=fee
                stop=entry-side*(stop_atr if side>0 else short_stop)*ea
                orders[oc]=np.array([i,q,o[i],fee,9,0.,q,cash]);oc+=1
        if q!=0:
            side=np.sign(q);price=stop;reason=2
            if q<0:
                capprice=cash/(abs(q)*(2+cost))*.995
                if capprice<price:price=capprice;reason=6
            trigger=liquid[i] and ((q>0 and l[i]<=price) or (q<0 and h[i]>=price))
            if trigger:price=min(o[i],price) if q>0 else max(o[i],price)

            adverse=h[i] if q>0 or not trigger else price
            wg=abs(q)*adverse/max(cash+q*adverse,1e-12);maxgross=max(maxgross,wg)
            if wg>1+1e-9:intrabr+=1
            if trigger:
                delta=-q;fee=abs(q)*price*cost;cash-=delta*price+fee;fees+=fee
                orders[oc]=np.array([i,delta,price,fee,reason,q,0.,cash]);oc+=1
                trades[tc]=np.array([born,i,side,cash-ep_base,fees,ecarry,(i-born)/96.]);tc+=1
                q=0.;until=i+cooldown
            elif i+1<end and (valid[i+1] if update_clock is None else update_clock[i+1]):
                completed_close=c[i] if update_prices is None else update_prices[i+1]
                profit=side*(completed_close-entry);threshold=lock_trigger if side>0 else short_lock
                if not locked and profit>=threshold*ea:
                    locked=True;floor=short_lock_profit if side<0 and short_lock_profit>=0 else lock_profit
                    level=entry+side*max(floor*ea,entry*2.2*cost)
                    stop=max(stop,level) if side>0 else min(stop,level)
                trailing=short_trail_atr if side<0 else trail_atr
                if trailing>0 and locked:
                    level=completed_close-side*trailing*ea;stop=max(stop,level) if side>0 else min(stop,level)
        stop_path[i]=stop if q!=0 else np.nan
        equity=cash+q*c[i];peak=max(peak,equity);eq[i]=equity;exp[i]=abs(q)*c[i]/max(equity,1e-12);cp[i]=cash;qp[i]=q;dd[i]=1-equity/peak;cy[i]=totalcarry
    return eq,exp,cp,qp,dd,cy,orders[:oc],trades[:tc],maxgross,gaps,intrabr,stop_path

def run(d,p,features=None,start=None,end=None,initial=INITIAL):

    if not (0<=p.short_cap<1 and 0<p.long_cap<=1 and p.cost>=0 and p.carry>=0 and p.vol_target>0):raise ValueError('Invalid configuration')
    f=features or Features(d);args,diagnostics=f.get(p)
    a=0 if start is None else int(d.index.searchsorted(pd.Timestamp(start,tz='UTC')))
    z=len(d) if end is None else int(d.index.searchsorted(pd.Timestamp(end,tz='UTC')))
    if z-a<2:raise ValueError('Empty window')
    x=simulate(*(d[k].to_numpy() for k in ['open','high','low','close','tradable']),*args,a,z,initial,p.cost,p.carry,p.stop_atr,p.lock_trigger,p.lock_profit,p.trail_atr,p.short_cap,p.short_stop,p.short_lock,p.dd_soft,p.dd_hard,p.cooldown_bars,p.max_hold_bars,f.update_clock,p.short_trail_atr,p.short_lock_profit,p.long_cap,f.update_prices)
    path=pd.DataFrame(dict(equity=x[0],exposure=x[1],cash=x[2],units=x[3],local_drawdown=x[4],carry=x[5]),index=d.index).iloc[a:z]
    path['standing_stop_next_bar']=x[11][a:z]
    path['signed_exposure']=path.units*d.close.reindex(path.index)/path.equity
    fills=pd.DataFrame(x[6],columns=['bar','delta_units','price','cost','reason','closed_units','result_units','cash_after']);fills['timestamp']=d.index[fills.bar.astype(int)]
    tr=pd.DataFrame(x[7],columns=['entry_bar','exit_bar','direction','pnl','cost','carry','days']);tr['entry_time']=d.index[tr.entry_bar.astype(int)];tr['exit_time']=d.index[tr.exit_bar.astype(int)]
    if abs(path.units.iloc[-1])<1e-8 and abs(path.equity.iloc[-1]-initial-tr.pnl.sum())>1e-5:raise AssertionError('Episode ledger mismatch')
    path['restricted_collateral']=2*abs(path.units.clip(upper=0))*d.close.reindex(path.index)
    path['free_cash']=path.cash-path.restricted_collateral
    audit=dict(max_observed_gross=float(x[8]),gap_cap_breaches=int(x[9]),intrabar_cap_breaches=int(x[10]),all_fills_tradable=bool(d.tradable.reindex(fills.timestamp).all()),terminal_flat=bool(abs(path.units.iloc[-1])<1e-8))
    return path,fills,tr,audit

def load_data(file):

    raw = pd.read_csv(file)
    tc = next((x for x in ('timestamp','open_time','time','datetime','date') if x in raw),None)
    if tc is None: raise ValueError('Missing timestamp column')
    idx = pd.to_datetime(raw[tc].astype(str),format='ISO8601',utc=True)
    if idx.duplicated().any(): raise ValueError('Duplicate timestamps')
    d = raw[['open','high','low','close','volume']].apply(pd.to_numeric,errors='raise')
    d.index = idx
    if not np.isfinite(d.to_numpy()).all(): raise ValueError('Nonfinite OHLCV')
    if (d[['open','high','low','close']]<=0).any().any() or (d.volume<0).any(): raise ValueError('Invalid prices/volume')
    if ((d.high<d[['open','close','low']].max(axis=1))|(d.low>d[['open','close','high']].min(axis=1))).any(): raise ValueError('OHLC envelope violation')
    filled = pd.Series(False,index=idx)
    if 'is_filled' in raw:
        flags = raw.is_filled.astype(str).str.lower()
        if not flags.isin(['true','false','0','1']).all(): raise ValueError('Invalid is_filled flag')
        filled = pd.Series(flags.isin(['true','1']).to_numpy(),index=idx)
    d['tradable'] = (d.volume>0)&~filled
    d = d.sort_index()
    offgrid = d.index != d.index.floor('15min')
    audit = dict(raw_bars=len(d),off_grid_excluded=int(offgrid.sum()),zero_volume=int((d.volume==0).sum()),filled_flag=int(filled.sum()))
    if offgrid.any(): raise ValueError('Off-grid timestamps: refusing silent exclusions')
    d = d.loc[~offgrid]
    if d.index.to_series().diff().dropna().median()!=pd.Timedelta('15min'): raise ValueError('Expected 15-minute data; refusing coarse-data upsampling')
    if not len(d): raise ValueError('No 15-minute bars')
    grid = pd.date_range(d.index[0],d.index[-1],freq='15min')
    d = d.reindex(grid)
    missing = d.close.isna()
    previous = d.close.ffill()
    for c in ('open','high','low','close'): d[c] = d[c].where(~missing,previous)
    d['volume'] = d.volume.fillna(0)
    d['tradable'] = d.tradable.eq(True)
    d.index.name = 'timestamp'
    audit.update(grid_bars=len(d),missing_synthetic=int(missing.sum()),untradable=int((~d.tradable).sum()),start=str(d.index[0]),end=str(d.index[-1]))
    d.attrs['audit'] = audit
    return d

def resample(d,rule,offset=None):
    b = d.resample(rule,origin='start_day',offset=offset).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum','tradable':'sum'})
    b.index += pd.Timedelta(rule)
    expected = int(pd.Timedelta(rule)/pd.Timedelta('15min'))
    b['complete'] = b.tradable==expected
    prev=b.close.shift()
    tr=pd.concat([b.high-b.low,(b.high-prev).abs(),(b.low-prev).abs()],axis=1).max(axis=1)
    b['atr']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    r=b.close.diff();up=r.clip(lower=0).ewm(alpha=1/3,adjust=False,min_periods=3).mean();dn=(-r.clip(upper=0)).ewm(alpha=1/3,adjust=False,min_periods=3).mean()
    b['rsi']=(100*up/(up+dn)).fillna(50)
    return b

def benchmark(d,cost=.0015,initial=INITIAL):
    ix=np.flatnonzero(d.tradable.to_numpy())
    if not len(ix) or not d.tradable.iloc[-1]: raise ValueError('Benchmark terminal open unavailable; no retrospective terminal fill')
    a,z=ix[0],len(d)-1
    qty=initial/(d.open.iloc[a]*(1+cost));out=pd.Series(initial,index=d.index,dtype=float)
    out.iloc[a:z+1]=qty*d.close.iloc[a:z+1]
    out.iloc[z:]=qty*d.open.iloc[z]*(1-cost)
    return out

def quarters(eq,bh,initial=INITIAL):
    rows=[];prev=prevbh=initial
    for period,s in eq.groupby(eq.index.tz_localize(None).to_period('Q')):
        b=bh.reindex(s.index);complete=s.index[0]<=period.start_time.tz_localize('UTC') and s.index[-1]>=period.end_time.floor('15min').tz_localize('UTC')
        sr=s.iloc[-1]/prev-1;br=b.iloc[-1]/prevbh-1
        rows.append(dict(quarter=str(period),strategy_return_pct=100*sr,buy_hold_return_pct=100*br,excess_pct=100*(sr-br),beat=bool(sr>br),complete=bool(complete)))
        prev=s.iloc[-1];prevbh=b.iloc[-1]
    return pd.DataFrame(rows)

def metrics(path,fills,tr,bh,audit,initial=INITIAL):

    eq=path.equity;days=(len(eq)*15/1440);daily=eq.resample('1D').last();r=daily.pct_change();r.iloc[0]=daily.iloc[0]/initial-1;r=r.fillna(0)
    downside=np.sqrt(np.mean(np.minimum(r,0)**2));sh=np.sqrt(365)*r.mean()/r.std(ddof=1) if r.std()>0 else 0.
    dd=eq/pd.Series(np.maximum.accumulate(np.r_[initial,eq.to_numpy()])[1:],index=eq.index)-1
    under=dd<-1e-10;longest=cur=0
    for u in under:cur=cur+1 if u else 0;longest=max(longest,cur)
    q=quarters(eq,bh,initial);qc=q[q.complete];wins=tr.loc[tr.pnl>0,'pnl'];loss=tr.loc[tr.pnl<=0,'pnl']
    m=dict(total_return_pct=float(100*(eq.iloc[-1]/initial-1)),short_trades=int((tr.direction<0).sum()),same_open_reversals=int(sum(len(g)==2 and g.iloc[0].closed_units*g.iloc[1].result_units<0 for _,g in fills.groupby('bar'))),cagr_pct=100*((eq.iloc[-1]/initial)**(365.25/days)-1),net_profit=float(eq.iloc[-1]-initial),final_equity=float(eq.iloc[-1]),sharpe=float(sh),sortino=float(np.sqrt(365)*r.mean()/downside) if downside>0 else 0.,max_drawdown_pct=float(-100*dd.min()),mean_exposure_pct=float(100*path.exposure.mean()),time_in_market_pct=float(100*(path.exposure>1e-6).mean()),longest_underwater_days=longest/96.,cost_paid=float(fills.cost.sum()),carry_paid=float(path.carry.iloc[-1]),executed_orders=len(fills),total_closed_trades=len(tr),win_rate_pct=float(100*(tr.pnl>0).mean()) if len(tr) else 0.,gross_profit=float(wins.sum()),gross_loss=float(-loss.sum()),average_winning_trade=float(wins.mean()) if len(wins) else 0.,average_losing_trade=float(loss.mean()) if len(loss) else 0.,largest_winning_trade=float(wins.max()) if len(wins) else 0.,largest_losing_trade=float(loss.min()) if len(loss) else 0.,average_holding_days=float(tr.days.mean()) if len(tr) else 0.,maximum_holding_days=float(tr.days.max()) if len(tr) else 0.,buy_hold_return_pct=float(100*(bh.iloc[-1]/initial-1)),quarter_beats=int(qc.beat.sum()),complete_quarters=len(qc),quarter_beat_pct=float(100*qc.beat.mean()) if len(qc) else 0.,daily_observations=len(r),**audit)
    gp=tr.pnl+tr.cost+tr.carry
    m.update(gross_profit_before_cost=float(gp.clip(lower=0).sum()),gross_loss_before_cost=float(-gp.clip(upper=0).sum()),pnl_before_cost_total=float(gp.sum()),annual_realized_vol_pct=float(r.std(ddof=1)*np.sqrt(365)*100),one_way_turnover_equity_pa=float((fills.delta_units.abs()*fills.price).sum()/eq.mean()/(days/365.25)),cost_pct_average_equity_pa=float(100*fills.cost.sum()/eq.mean()/(days/365.25)),short_episode_net_pnl=float(tr.loc[tr.direction<0,'pnl'].sum()),long_episode_net_pnl=float(tr.loc[tr.direction>0,'pnl'].sum()))
    return m,q,r

def export(dest,d,p,path,fills,tr,audit,label,initial=INITIAL):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);bh=benchmark(d.reindex(path.index),p.cost,initial)
    m,q,r=metrics(path,fills,tr,bh,audit,initial);m['result_type']=label
    path.assign(buy_hold=bh).to_csv(dest/'equity.csv');fills.to_csv(dest/'fills.csv',index=False);tr.to_csv(dest/'trades.csv',index=False);q.to_csv(dest/'quarters.csv',index=False)
    (dest/'metrics.json').write_text(json.dumps(m,indent=2));return m


FROZEN_PARAMETERS = {'rule': '4h', 'offset_hours': 0, 'trend_mode': 0, 'momentum_days': 40, 'entry_threshold': 0.66, 'exit_threshold': 0.0, 'vol_fast_days': 7, 'vol_slow_days': 30, 'regime_days': 40, 'breakout': 36, 'exit_bars': 54, 'stop_atr': 4.0, 'lock_trigger': 1.25, 'lock_profit': 1.0, 'trail_atr': 0.0, 'vol_rank_max': 0.9, 'vol_target': 0.8, 'weekends': False, 'long_cap': 1.0, 'short_hysteresis': 0.01, 'short_require_long_bear': True, 'short_mode': 1, 'short_cap': 0.8, 'short_breakout': 72, 'short_exit': 18, 'short_stop': 3, 'short_lock': 1.5, 'short_vol_rank': 1.01, 'short_regime_days': 75, 'short_slope_days': 30, 'short_regime_exit': True, 'short_trail_atr': 4.0, 'short_lock_profit': 1.0, 'short_vol_target': 0.65, 'short_signal_mode': 0, 'short_rally_ema': 20, 'short_shock_atr': 0.0, 'short_shock_bars': 6, 'only_shorts': False, 'regime_exit': False, 'hurst_min': 0.0, 'long_adx_min': 0.0, 'short_adx_min': 0, 'short_hurst_min': 0.45, 'channel_close': False, 'pullback_rsi': 0.0, 'dd_soft': 0.15, 'dd_hard': 0.26, 'cooldown_bars': 96, 'max_hold_bars': 5760, 'cost': 0.0015, 'carry': 0.05, 'delay': 0, 'long_short': True}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True);ap.add_argument('--config');ap.add_argument('--start');ap.add_argument('--end')
    ap.add_argument('--out',default='bitcoin_crash_results');ap.add_argument('--long-only',action='store_true')
    ap.add_argument('--short-only',action='store_true')
    ap.add_argument('--cost',type=float);ap.add_argument('--carry',type=float);ap.add_argument('--delay',type=int)
    args=ap.parse_args();p=Config(**(json.loads(Path(args.config).read_text())['parameters'] if args.config else FROZEN_PARAMETERS))
    for k in ['cost','carry','delay']:
        if getattr(args,k) is not None:p=replace(p,**{k:getattr(args,k)})
    if args.long_only and args.short_only:ap.error('Choose only one directional mode')
    if args.long_only:p=replace(p,long_short=False,only_shorts=False)
    if args.short_only:p=replace(p,long_short=True,only_shorts=True)
    if p.cost<.0015 or p.delay<0 or p.carry<0:ap.error('Cost must be >=0.15% per side; delay and carry nonnegative')
    np.random.seed(26);d=load_data(args.data);x=run(d,p,start=args.start,end=args.end);path,fills,tr,audit=x
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if audit['gap_cap_breaches'] or audit['intrabar_cap_breaches'] or (path.equity<=0).any() or not audit['terminal_flat']:
        (out/'rejection_audit.json').write_text(json.dumps(audit,indent=2));path.to_csv(out/'rejected_equity.csv')
        raise SystemExit('Replay rejected: observed >1x gross exposure or nonpositive equity. No clipping of losses.')
    m=export(out,d,p,*x,'historical replay; parameters selected using supplied history')
    (out/'used_config.json').write_text(json.dumps(asdict(p),indent=2));(out/'data_audit.json').write_text(json.dumps(d.attrs['audit'],indent=2));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
