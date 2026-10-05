from dataclasses import dataclass, asdict, replace
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    regime_days: int=35
    breakout: int=10
    exit_bars: int=36
    stop_atr: float=2.
    lock_trigger: float=2.5
    lock_profit: float=.5
    trail_atr: float=6.
    vol_rank_max: float=.9
    vol_target: float=1.2
    weekends: bool=True
    short_mode: int=1
    short_cap: float=.65
    short_breakout: int=36
    short_exit: int=18
    short_stop: float=2.5
    short_lock: float=1.5
    short_vol_rank: float=1.01
    short_regime_days: int=0
    short_slope_days: int=10
    short_regime_exit: bool=True
    short_trail_atr: float=4.
    short_lock_profit: float=1.
    short_vol_target: float=0.
    short_signal_mode: int=0
    short_shock_atr: float=0.
    short_shock_bars: int=6
    only_shorts: bool=False
    regime_exit: bool=False
    hurst_min: float=0.
    long_adx_min: float=20
    short_adx_min: float=0.
    short_hurst_min: float=0.
    channel_close: bool=False
    pullback_rsi: float=0.
    dd_soft: float=.25
    dd_hard: float=.3
    cooldown_bars: int=2
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
        bull=(self.day.close>ma).reindex(b.index,method='ffill',fill_value=False)
        sma=self.day.close.rolling(p.short_regime_days or p.regime_days).mean()
        bear=((self.day.close<sma)&(sma<sma.shift(p.short_slope_days))).reindex(b.index,method='ffill',fill_value=False)
        if p.short_shock_atr>0:
            fast=b.close.ewm(span=12,adjust=False).mean()
            shock=(b.close-b.close.shift(p.short_shock_bars)<-p.short_shock_atr*b.atr.shift(p.short_shock_bars))&(b.close<fast)&(fast<fast.shift(3))
            bear= bear|shock
        en=((b.close>(b.close if p.channel_close else b.high).shift().rolling(p.breakout).max())|((b.rsi<p.pullback_rsi)&(p.pullback_rsi>0)))&bull&(b['rank']<=p.vol_rank_max)&(b.hurst>=p.hurst_min)&(b.adx>=p.long_adx_min)
        if not p.weekends:en&=b.index.dayofweek<5
        sn=(b.close<b.low.shift().rolling(p.short_breakout).min())&bear&(b['rank']<=p.short_vol_rank)&(b.hurst>=p.short_hurst_min)&(b.adx>=p.short_adx_min)
        if p.short_signal_mode==1:sn=(b.close<b.close.ewm(span=p.short_breakout,adjust=False).mean())&(b.close<b.close.shift(3))&bear&(b['rank']<=p.short_vol_rank)&(b.hurst>=p.short_hurst_min)&(b.adx>=p.short_adx_min)
        if p.short_mode==2:sn&=b.close<b.close.ewm(span=20,adjust=False).mean()
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
        frame=pd.DataFrame(dict(en=en,sn=sn,lx=lx,sx=sx,valid=b.complete,atr=b.atr,size=pd.Series(np.where(en,p.vol_target,p.short_vol_target or p.vol_target),index=b.index).div(b.vol).clip(upper=1),hurst=b.hurst,vol=b.vol,vol_rank=b['rank'],adx=b.adx,bull=bull,bear=bear,short_break=(b.close<b.low.shift().rolling(p.short_breakout).min()),short_below_ma=(self.day.close<sma).reindex(b.index,method='ffill',fill_value=False),short_slope=(sma<sma.shift(p.short_slope_days)).reindex(b.index,method='ffill',fill_value=False)))
        aligned=frame.reindex(self.d.index)
        if p.delay:aligned=aligned.shift(p.delay)
        self.update_clock=aligned.valid.eq(True).to_numpy(bool)
        pending=None
        arr=aligned.to_numpy(copy=True); vi=aligned.columns.get_loc('valid')
        liquid=self.d.tradable.to_numpy()
        for i in range(len(arr)):
            if arr[i,vi] is True or arr[i,vi]==True: pending=arr[i].copy()
            if pending is not None:
                if liquid[i]: arr[i]=pending;pending=None
                else: arr[i,vi]=False
        aligned=pd.DataFrame(arr,index=aligned.index,columns=aligned.columns)
        flags=[aligned[k].eq(True).to_numpy(bool) for k in ['en','sn','lx','sx','valid']]
        vals=[pd.to_numeric(aligned[k]).fillna(0).to_numpy(float) for k in ['atr','size']]
        return (*flags,*vals),frame

@njit(cache=True)
def simulate(o,h,l,c,liquid,en,sn,lx,sx,valid,atr,size,start,end,initial,cost,carry,stop_atr,lock_trigger,lock_profit,trail_atr,short_cap,short_stop,short_lock,ddsoft,ddhard,cooldown,maxhold,update_clock=None,short_trail_atr=0.,short_lock_profit=-1.):
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
                fraction=min(1.,size[i]) if side>0 else min(short_cap,size[i])
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
                profit=side*(c[i]-entry);threshold=lock_trigger if side>0 else short_lock
                if not locked and profit>=threshold*ea:
                    locked=True;floor=short_lock_profit if side<0 and short_lock_profit>=0 else lock_profit
                    level=entry+side*max(floor*ea,entry*2.2*cost)
                    stop=max(stop,level) if side>0 else min(stop,level)
                trailing=short_trail_atr if side<0 else trail_atr
                if trailing>0 and locked:
                    level=c[i]-side*trailing*ea;stop=max(stop,level) if side>0 else min(stop,level)
        stop_path[i]=stop if q!=0 else np.nan
        equity=cash+q*c[i];peak=max(peak,equity);eq[i]=equity;exp[i]=abs(q)*c[i]/max(equity,1e-12);cp[i]=cash;qp[i]=q;dd[i]=1-equity/peak;cy[i]=totalcarry
    return eq,exp,cp,qp,dd,cy,orders[:oc],trades[:tc],maxgross,gaps,intrabr,stop_path

def run(d,p,features=None,start=None,end=None,initial=INITIAL):
    if not (0<=p.short_cap<1 and p.cost>=0 and p.carry>=0 and p.vol_target>0):raise ValueError('Invalid configuration')
    f=features or Features(d);args,diagnostics=f.get(p)
    a=0 if start is None else int(d.index.searchsorted(pd.Timestamp(start,tz='UTC')))
    z=len(d) if end is None else int(d.index.searchsorted(pd.Timestamp(end,tz='UTC')))
    if z-a<2:raise ValueError('Empty window')
    x=simulate(*(d[k].to_numpy() for k in ['open','high','low','close','tradable']),*args,a,z,initial,p.cost,p.carry,p.stop_atr,p.lock_trigger,p.lock_profit,p.trail_atr,p.short_cap,p.short_stop,p.short_lock,p.dd_soft,p.dd_hard,p.cooldown_bars,p.max_hold_bars,f.update_clock,p.short_trail_atr,p.short_lock_profit)
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
    big=[]
    if missing.any():
        g=(missing!=missing.shift()).cumsum()
        for _,x in missing.groupby(g):
            if x.iloc[0] and len(x)>=7*96:big.append([str(x.index[0]),str(x.index[-1])])
    audit.update(big_gaps=big,grid_bars=len(d),missing_synthetic=int(missing.sum()),untradable=int((~d.tradable).sum()),start=str(d.index[0]),end=str(d.index[-1]))
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

FROZEN_PARAMETERS=asdict(Config())

WINDOWS = [
    ("2017_2020", None, "2021-01-01"),
    ("2021_2025", "2021-01-01", "2026-01-01"),
    ("2026_to_date", "2026-01-01", None),
    ("full", None, None),
]


def bh_stats(bh):
    daily = bh.resample("1D").last()
    r = daily.pct_change().fillna(0)
    years = len(bh) * 15 / 1440 / 365.25
    return {
        "bh_cagr_pct": float(((bh.iloc[-1] / bh.iloc[0]) ** (1 / years) - 1) * 100),
        "bh_sharpe": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0,
        "bh_max_drawdown_pct": float(-(bh / bh.cummax() - 1).min() * 100),
    }


def plot_window(dest, d, path, tr, label):
    bh = INITIAL * d.close.reindex(path.index) / d.open.reindex(path.index).iloc[0]
    eq = path.equity
    fig, ax = plt.subplots(4, 1, figsize=(14, 15), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2, 2.2]})
    ax[0].plot(eq.index, eq.values, lw=1.2, label="Strategy")
    ax[0].plot(bh.index, bh.values, lw=0.9, alpha=0.7, label="Buy & hold")
    ax[0].set_yscale("log")
    ax[0].set_title(f"ETHUSDT {label}: equity vs buy & hold (log, 0.15% per side, no leverage)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].fill_between(eq.index, (eq / eq.cummax() - 1) * 100, 0, color="tab:red", alpha=0.5, label="Strategy")
    ax[1].plot(bh.index, (bh / bh.cummax() - 1) * 100, color="gray", lw=0.7, label="Buy & hold")
    ax[1].set_ylabel("Drawdown %")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    ax[2].fill_between(path.index, path.signed_exposure.values, 0, color="tab:purple", alpha=0.6, step="post")
    ax[2].set_ylabel("Signed exposure")
    ax[2].set_ylim(-1.05, 1.05)
    ax[2].grid(alpha=0.3)
    px = d.close.reindex(path.index)
    ax[3].plot(px.index, px.values, color="black", lw=0.5)
    ax[3].set_yscale("log")
    if len(tr):
        for side, col, mk, nm in ((1, "tab:green", "^", "Long entry"), (-1, "tab:red", "v", "Short entry")):
            t = tr[tr.direction == side]
            ax[3].scatter(t.entry_time, d.open.reindex(t.entry_time).values, s=14, color=col, marker=mk, label=nm)
    ax[3].set_title("Trade history")
    ax[3].legend()
    ax[3].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(dest) / "backtest.png", dpi=105)
    plt.close(fig)


def run_window(d, f, p, name, a, b, out):
    path, fills, tr, audit = run(d, p, features=f, start=a, end=b)
    if audit["gap_cap_breaches"] or audit["intrabar_cap_breaches"] or (path.equity <= 0).any() or not audit["terminal_flat"]:
        (out / f"rejection_audit_{name}.json").write_text(json.dumps(audit, indent=2))
        raise SystemExit(f"Replay rejected for {name}: observed >1x gross exposure or nonpositive equity.")
    dest = out / name
    m = export(dest, d, p, path, fills, tr, audit, name)
    bh = benchmark(d.reindex(path.index), p.cost)
    m.update(bh_stats(bh))
    (dest / "metrics.json").write_text(json.dumps(m, indent=2))
    plot_window(dest, d, path, tr, name)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="eth_results")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--cost", type=float)
    ap.add_argument("--carry", type=float)
    ap.add_argument("--delay", type=int)
    args = ap.parse_args()
    p = Config(**FROZEN_PARAMETERS)
    for k in ["cost", "carry", "delay"]:
        if getattr(args, k) is not None:
            p = replace(p, **{k: getattr(args, k)})
    if args.long_only:
        p = replace(p, long_short=False)
    if p.cost < .0015 or p.delay < 0 or p.carry < 0:
        ap.error("Cost must be >=0.15% per side; delay and carry nonnegative")
    np.random.seed(26)
    d = load_data(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit = d.attrs["audit"]
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2))
    (out / "used_config.json").write_text(json.dumps(asdict(p), indent=2))
    f = Features(d)
    wins = [("custom", args.start, args.end)] if (args.start or args.end) else list(WINDOWS)
    expanded = []
    for name, a, b in wins:
        expanded.append((name, a, b))
        lo = d.index[0] if a is None else pd.Timestamp(a, tz="UTC")
        hi = d.index[-1] + pd.Timedelta("15min") if b is None else pd.Timestamp(b, tz="UTC")
        for gs, ge in audit["big_gaps"]:
            gs = pd.Timestamp(gs)
            if lo < gs < hi:
                expanded.append((f"{name}_before_data_gap", a, gs.strftime("%Y-%m-%d %H:%M")))
    rows = {}
    for name, a, b in expanded:
        rows[name] = run_window(d, f, p, name, a, b, out)
    keys = ["total_return_pct", "cagr_pct", "sharpe", "sortino", "max_drawdown_pct", "win_rate_pct", "total_closed_trades", "short_trades", "average_holding_days", "quarter_beats", "complete_quarters", "quarter_beat_pct", "buy_hold_return_pct", "bh_cagr_pct", "bh_sharpe", "bh_max_drawdown_pct", "cost_paid", "carry_paid", "max_observed_gross"]
    summary = pd.DataFrame({k: {x: rows[k][x] for x in keys} for k in rows}).T
    summary.to_csv(out / "summary.csv")
    (out / "summary.json").write_text(json.dumps({k: {x: rows[k][x] for x in keys} for k in rows}, indent=2))
    print(summary.round(2).T.to_string())
    if audit["big_gaps"]:
        print("Data gaps of 7 days or more (untradable, no fills):", audit["big_gaps"])


if __name__ == "__main__":
    main()
